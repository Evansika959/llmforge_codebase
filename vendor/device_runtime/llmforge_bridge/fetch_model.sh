#!/usr/bin/env bash
# Download the example model weights from a URL where a copy is hosted.
#
# The weights (~138 MB) are not part of the repository. Regenerating them from a checkpoint with
# llmforge_bridge/export_llmforge.py needs no download at all.
#
# Usage:
#   MODEL_URL="https://<host>/<path>/smollm2_135M.q8.bin" llmforge_bridge/fetch_model.sh
#   # or pass the URL as the first arg:
#   llmforge_bridge/fetch_model.sh <URL>
#
# The tokenizer (tokenizer_gpt2.bin) IS committed in git (small), so you only need the weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/smollm2_135M/smollm2_135M.q8.bin"
URL="${1:-${MODEL_URL:-}}"

if [ -z "$URL" ]; then
  echo "No model URL given. Set MODEL_URL or pass it as an argument." >&2
  echo "Alternatively, regenerate it from a checkpoint:" >&2
  echo "  python llmforge_bridge/export_llmforge.py <ckpt_dir> $DEST --version 2" >&2
  exit 1
fi

echo "downloading model -> $DEST"
curl -fL --retry 3 -o "$DEST" "$URL"

# sanity: real weights start with the 'ak42' magic, not an HTML/error page
if head -c 4 "$DEST" | grep -q "ak42"; then
  echo "OK: $(ls -lh "$DEST" | awk '{print $5}') downloaded."
else
  echo "WARNING: downloaded file doesn't look like a valid model (missing 'ak42' magic). Check the URL." >&2
  exit 1
fi
