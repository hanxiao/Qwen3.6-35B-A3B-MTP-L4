#!/usr/bin/env bash
# Spin up a SPOT NVIDIA L4 (g2-standard-8) serving Qwen3.6-35B-A3B-MTP with the
# llama.cpp Web UI + Prometheus telemetry, and block until it's ready.
#
# Uses your active gcloud project + auth (gcloud config / CLOUDSDK_CORE_PROJECT).
# Zones are enumerated live from the GCP API (every zone that offers the VM),
# US first so the Web UI stays low-latency. Override any of these via env:
#   ZONES="zone-a zone-b ..."   INSTANCE=name   MACHINE=g2-standard-8
set -euo pipefail

INSTANCE="${INSTANCE:-qwen36-mtp-l4-spot}"
MACHINE="${MACHINE:-g2-standard-8}"

# Every zone that offers $MACHINE, US first. No hardcoded zone list to rot.
if [ -z "${ZONES:-}" ]; then
  ALL="$(gcloud compute machine-types list --filter="name=$MACHINE" --format='value(zone)')"
  ZONES="$(printf '%s\n' "$ALL" | grep '^us-'; printf '%s\n' "$ALL" | grep -v '^us-')"
fi

say(){ printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

# What the VM runs on boot: disable ECC (+~10% bandwidth, lossless, one reboot),
# install Docker (the CUDA base image ships the driver + nvidia-container-toolkit
# but not docker), then serve. llama.cpp pulls the GGUF itself into the mounted
# cache, so it survives container restarts and spot preemption.
STARTUP="$(mktemp)"; cat > "$STARTUP" <<'SH'
#!/bin/bash
set -e
exec >>/var/log/qwen-startup.log 2>&1
echo "[startup] $(date -u)"
if nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | grep -qi Enabled; then
  echo "[startup] disabling ECC + rebooting"; nvidia-smi -e 0 || true; reboot; exit 0
fi
if ! command -v docker >/dev/null; then
  echo "[startup] installing docker"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi
echo "[startup] launching server"
mkdir -p /opt/models
docker rm -f llama-server 2>/dev/null || true
docker run -d --name llama-server --restart unless-stopped --gpus all -p 8080:8080 \
  -e LLAMA_CACHE=/models -v /opt/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --hf-repo unsloth/Qwen3.6-35B-A3B-MTP-GGUF --hf-file Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
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
[ -n "$ZONE" ] || { say "No spot L4 capacity in any offering zone"; exit 1; }

IP="$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
say "Created in $ZONE (IP $IP). Waiting for readiness — ECC reboot + docker install + ~22GB model download + load (~6-10 min) ..."
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
