#!/usr/bin/env bash
# Provision a SPOT NVIDIA L4 (g2-standard-8) serving Qwen3.6-35B-A3B-MTP with the
# llama.cpp Web UI + Prometheus telemetry, and block until it's ready.
#
# Cold-start optimized (~3-5 min vs ~10): the 22 GB GGUF is pulled from a same-region
# GCS bucket (sliced parallel download, ~60 s) instead of HuggingFace, and the docker
# install + image pull run concurrently with the model fetch. ECC-off reboot is kept
# (load-bearing: frees the VRAM that lets --ctx-size 56320 fit). HF is the fallback.
#
# Uses your active gcloud project + auth. Zones are enumerated live (US first).
# Override via env: ZONES="zone-a ..."  INSTANCE=name  MACHINE=g2-standard-8
#   GCS_MODEL=gs://bucket/path/model.gguf   (same-region bucket; see scripts/stage-model-gcs.sh)
set -euo pipefail

INSTANCE="${INSTANCE:-qwen36-mtp-l4-spot}"
MACHINE="${MACHINE:-g2-standard-8}"
GCS_MODEL="${GCS_MODEL:-gs://jinaai-dev-qwen36-mtp-l4/model.gguf}"

# Every zone that offers $MACHINE, US first (the GCS bucket is us-central1, so a US
# landing keeps the model read in-region). No hardcoded zone list to rot.
if [ -z "${ZONES:-}" ]; then
  ALL="$(gcloud compute machine-types list --filter="name=$MACHINE" --format='value(zone)')"
  ZONES="$(printf '%s\n' "$ALL" | grep '^us-'; printf '%s\n' "$ALL" | grep -v '^us-')"
fi

say(){ printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

# Optional: a custom image with docker + the llama.cpp image pre-baked removes the
# ~168 s install/pull from cold start (build once via scripts/build-tooling-image.sh).
# Auto-detected; falls back to the stock CUDA base image when absent.
TOOLING_FAMILY="${TOOLING_FAMILY:-qwen36-mtp-l4-tooling}"
PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [ -n "$PROJECT" ] && gcloud compute images describe-from-family "$TOOLING_FAMILY" --project="$PROJECT" >/dev/null 2>&1; then
  IMG=(--image-family="$TOOLING_FAMILY" --image-project="$PROJECT")
  say "Using pre-baked tooling image: $TOOLING_FAMILY (docker + llama.cpp image baked)"
else
  IMG=(--image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release)
fi

# Boot script: (1) disable ECC + reboot once; (2) on the 2nd boot, fetch the model and
# prepare docker IN PARALLEL, then serve from the local file. Logs phase timestamps to
# /var/log/qwen-startup.log. GCS_MODEL is substituted in (not a heredoc var) so the bucket
# path travels with the instance metadata.
STARTUP="$(mktemp)"; cat > "$STARTUP" <<SH
#!/bin/bash
set -e
exec >>/var/log/qwen-startup.log 2>&1
echo "[startup] boot \$(date -u +%H:%M:%S)"
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  echo "[startup] disabling ECC + rebooting"; nvidia-smi -e 0 || true; reboot; exit 0
fi
echo "[startup] ECC off — parallel provision \$(date -u +%H:%M:%S)"
mkdir -p /opt/models
MODEL=/opt/models/model.gguf

# (a) docker engine + server image
(
  if ! command -v docker >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  fi
  docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1 || docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
  echo "[startup] docker+image ready \$(date -u +%H:%M:%S)"
) &
DPID=\$!

# (b) the 22 GB model: same-region GCS (fast), HuggingFace as fallback
(
  if [ ! -f "\$MODEL" ]; then
    if gcloud storage cp "$GCS_MODEL" "\$MODEL"; then
      echo "[startup] model from GCS \$(date -u +%H:%M:%S)"
    else
      echo "[startup] GCS miss -> HF fallback \$(date -u +%H:%M:%S)"
      pip install -q -U "huggingface_hub[hf_xet]"
      HF_XET_HIGH_PERFORMANCE=1 python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("unsloth/Qwen3.6-35B-A3B-MTP-GGUF","Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",local_dir="/opt/models")
PY
      ln -sf /opt/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf "\$MODEL"
    fi
  fi
) &
MPID=\$!

wait \$DPID; wait \$MPID
echo "[startup] launching server \$(date -u +%H:%M:%S)"
docker rm -f llama-server 2>/dev/null || true
docker run -d --name llama-server --restart unless-stopped --gpus all -p 8080:8080 \
  -v /opt/models:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/model.gguf \
  --alias Qwen3.6-35B-A3B-Q4KXL-MTP --host 0.0.0.0 --port 8080 --jinja --tools all \
  --ctx-size 56320 --parallel 1 --flash-attn on -ngl 99 --n-cpu-moe 0 -ub 64 -b 512 \
  --no-mmap --threads 8 --spec-type draft-mtp --spec-draft-n-max 2 --no-warmup --metrics
echo "[startup] done \$(date -u +%H:%M:%S)"
SH

say "Opening firewall for :8080 (Web UI + API + /metrics) ..."
gcloud compute firewall-rules create allow-llama-8080 \
  --allow=tcp:8080 --target-tags=llama-server --source-ranges=0.0.0.0/0 \
  --description="llama.cpp web UI + API + metrics" 2>/dev/null || true

ZONE=""
for z in $ZONES; do
  say "Requesting SPOT $MACHINE (1x L4) in $z ..."
  if gcloud compute instances create "$INSTANCE" \
      --zone="$z" --machine-type="$MACHINE" \
      --provisioning-model=SPOT --instance-termination-action=STOP \
      --maintenance-policy=TERMINATE \
      "${IMG[@]}" \
      --boot-disk-size=80GB --boot-disk-type=pd-ssd \
      --tags=llama-server \
      --metadata-from-file=startup-script="$STARTUP"; then
    ZONE="$z"; break
  fi
  say "No spot capacity in $z, trying next ..."
done
rm -f "$STARTUP"
[ -n "$ZONE" ] || { say "No spot L4 capacity in any offering zone"; exit 1; }

IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
say "Created in $ZONE (IP $IP). Waiting for readiness — ECC reboot + parallel(docker, ~22GB GCS pull) + load (~3-5 min) ..."
until curl -fsS --max-time 5 "http://$IP:8080/health" 2>/dev/null | grep -q '"status":"ok"'; do
  printf '.'; sleep 10
done

say "READY (spot — may be preempted anytime; the container auto-resumes via --restart unless-stopped)"
cat <<EOF
  Web UI:   http://$IP:8080
  API:      http://$IP:8080/v1/chat/completions
  Metrics:  http://$IP:8080/metrics   (Prometheus)
  Logs:     gcloud compute ssh $INSTANCE --zone=$ZONE --command 'sudo docker logs -f llama-server'
  Stop:     gcloud compute instances stop   $INSTANCE --zone=$ZONE
  Delete:   gcloud compute instances delete $INSTANCE --zone=$ZONE
EOF
