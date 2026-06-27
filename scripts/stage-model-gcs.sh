#!/usr/bin/env bash
# One-time: stage the Qwen3.6-35B-A3B-MTP Q4_K_XL GGUF into a same-region GCS bucket so
# provision-spot.sh pulls it fast (sliced parallel download, ~60s) instead of from
# HuggingFace (~5-8 min). This is what makes the cold start ~3-5 min instead of ~10.
#
# Run it where you have (a) gcloud auth that can WRITE to the bucket and (b) the GGUF on
# disk (SRC=...) or bandwidth to fetch it from HF. Fastest path is on a GCE box in the
# bucket's region with --scopes=cloud-platform (GCE->GCS stays on Google's network).
#
# Env: GCS_MODEL=gs://bucket/path/model.gguf   SRC=/path/to/local.gguf (optional)
set -euo pipefail
GCS_MODEL="${GCS_MODEL:-gs://jinaai-dev-qwen36-mtp-l4/model.gguf}"
SRC="${SRC:-}"
BUCKET="gs://$(printf '%s' "$GCS_MODEL" | sed -E 's#^gs://([^/]+)/.*#\1#')"

if [ -z "$SRC" ]; then
  echo "▶ fetching GGUF from HuggingFace (pass SRC=/path/to.gguf to skip) ..."
  pip install -q -U "huggingface_hub[hf_xet]"
  SRC="$(HF_XET_HIGH_PERFORMANCE=1 python3 - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
                      "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", local_dir="/tmp/qwen-gguf"))
PY
)"
fi

echo "▶ staging $(du -h "$SRC" 2>/dev/null | cut -f1) -> $GCS_MODEL"
gcloud storage cp "$SRC" "$GCS_MODEL"      # sliced parallel upload by default
gcloud storage ls -l "$GCS_MODEL"

cat <<EOF

Grant the deploy VMs' service account read on the bucket (one-time):
  gcloud storage buckets add-iam-policy-binding $BUCKET \\
    --member=serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com \\
    --role=roles/storage.objectViewer
The default GCE storage scope (read-only) is enough for provision-spot.sh to read it.
EOF
