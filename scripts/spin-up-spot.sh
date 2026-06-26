#!/usr/bin/env bash
# Spin up a SPOT NVIDIA L4 (g2-standard-8) serving Qwen3.6-35B-A3B-MTP with the
# llama.cpp Web UI + Prometheus telemetry, and block until it's ready.
#
# Uses your active gcloud project + auth (gcloud config / CLOUDSDK_CORE_PROJECT).
# Override via env: ZONES, INSTANCE, MACHINE.
set -euo pipefail

ZONES="${ZONES:-us-west1-a us-central1-a us-east1-d europe-west4-a}"
INSTANCE="${INSTANCE:-qwen36-mtp-l4-spot}"
MACHINE="${MACHINE:-g2-standard-8}"

say(){ printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

# What the VM runs on boot: disable ECC (+10% bandwidth, one reboot), fetch the
# model once, then serve with the built-in Web UI + Prometheus telemetry (--metrics).
STARTUP="$(mktemp)"; cat > "$STARTUP" <<'SH'
#!/bin/bash
set -e
exec >>/var/log/qwen-startup.log 2>&1
echo "[startup] $(date -u)"
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  echo "[startup] disabling ECC + rebooting"; nvidia-smi -e 0 || true; reboot; exit 0
fi
DIR=/opt/models/Qwen3.6-35B-A3B-MTP-GGUF
if [ ! -f "$DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" ]; then
  echo "[startup] downloading model"; mkdir -p "$DIR"
  pip install -q -U "huggingface_hub[hf_xet]"
  HF_XET_HIGH_PERFORMANCE=1 python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("unsloth/Qwen3.6-35B-A3B-MTP-GGUF","Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
               local_dir="/opt/models/Qwen3.6-35B-A3B-MTP-GGUF")
PY
fi
echo "[startup] launching server"
docker rm -f llama-server 2>/dev/null || true
docker run -d --name llama-server --restart unless-stopped --gpus all -p 8080:8080 \
  -v /opt/models:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --alias Qwen3.6-35B-A3B-Q4KXL-MTP --host 0.0.0.0 --port 8080 --jinja --tools all \
  --ctx-size 8192 --parallel 1 --flash-attn on -ngl 99 --n-cpu-moe 0 -ub 64 -b 512 \
  --no-mmap --threads 8 --spec-type draft-mtp --spec-draft-n-max 2 --metrics
echo "[startup] done"
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
      --image-family=common-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=120GB --boot-disk-type=pd-balanced \
      --tags=llama-server \
      --metadata-from-file=startup-script="$STARTUP"; then
    ZONE="$z"; break
  fi
  say "No spot capacity in $z, trying next ..."
done
rm -f "$STARTUP"
[ -n "$ZONE" ] || { say "No spot L4 capacity in any of: $ZONES"; exit 1; }

IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
say "Created in $ZONE (IP $IP). Waiting for readiness — ECC reboot + ~22GB model download + load (~5-9 min) ..."
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
