#!/usr/bin/env bash
# Fetch the pinned third-party checkouts into third_party/ (git-ignored).
#
#   ReaLLM-Forge    model code the ZEUS GPU target instantiates (llmforge.hw.zeus)
#   nanollmforge.c  on-device inference runtime used by the device measurement harness
#
# Override the sources with REALLM_FORGE_URL and NANOLLMFORGE_C_URL. The nanollmforge.c repository
# URL is withheld during anonymous review, so that checkout is skipped until the variable is set.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/third_party"

fetch() {
  local name="$1" url="$2" commit="$3" dir="$ROOT/third_party/$1"
  if [ -z "$url" ]; then
    echo "skip $name: no repository URL set"
    return
  fi
  [ -d "$dir/.git" ] || git clone --quiet "$url" "$dir"
  git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$commit"
  echo "$name at $(git -C "$dir" rev-parse --short HEAD)"
}

fetch ReaLLM-Forge "${REALLM_FORGE_URL:-https://github.com/ReaLLMASIC/ReaLLM-Forge.git}" \
  0237c4d5c40a6182864242564c6e61627f7399ef
fetch nanollmforge.c "${NANOLLMFORGE_C_URL:-}" d517dd8234ba723e7ec394d708a0347218381e9f
