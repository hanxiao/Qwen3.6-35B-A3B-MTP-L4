#!/usr/bin/env bash
# Build the OPTIONAL pre-baked tooling image that provision-spot.sh auto-detects.
# It's the stock CUDA base image + docker + nvidia-ctk configured + the llama.cpp
# server image pre-pulled. NO model is baked (the model stays in GCS, updated
# independently). This removes the ~168 s docker-install + image-pull from cold
# start (~3.5 min -> ~2 min). Rebuild only when you bump the llama.cpp image.
#
# Runs on a cheap CPU box (no GPU needed — the driver files are baked from the base
# image and only load at runtime on the g2). Env: ZONE FAMILY IMAGE BUILDER IMAGE_LOCATION
set -euo pipefail
ZONE="${ZONE:-us-central1-a}"
FAMILY="${FAMILY:-qwen36-mtp-l4-tooling}"
IMAGE="${IMAGE:-$FAMILY-$(date +%Y%m%d)}"
BUILDER="${BUILDER:-qwen-tooling-builder}"

BAKE="$(mktemp)"; cat > "$BAKE" <<'SH'
#!/bin/bash
set -e
exec >>/var/log/bake.log 2>&1
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
echo "BAKE_DONE $(date -u)"
SH

echo "▶ builder $BUILDER in $ZONE"
gcloud compute instances create "$BUILDER" --zone="$ZONE" --machine-type=e2-standard-4 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=60GB --metadata-from-file=startup-script="$BAKE"
rm -f "$BAKE"

echo "▶ waiting for bake (docker + image pull) ..."
until gcloud compute ssh "$BUILDER" --zone="$ZONE" \
        --command='grep -q BAKE_DONE /var/log/bake.log' \
        -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no 2>/dev/null; do printf '.'; sleep 12; done

echo "▶ stop + create image $IMAGE (family $FAMILY)"
gcloud compute instances stop "$BUILDER" --zone="$ZONE"
gcloud compute images create "$IMAGE" \
  --source-disk="$BUILDER" --source-disk-zone="$ZONE" \
  --family="$FAMILY" --storage-location="${IMAGE_LOCATION:-us}"
gcloud compute instances delete "$BUILDER" --zone="$ZONE" --quiet
echo "▶ done: image $IMAGE in family $FAMILY — provision-spot.sh will auto-detect it."
