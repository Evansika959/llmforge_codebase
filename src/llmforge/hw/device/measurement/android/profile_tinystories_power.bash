#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
# The inference runtime comes from llmforge.paths.DEVICE_RUNTIME, which LLMFORGE_DEVICE_RUNTIME
# overrides. Downloads and build products go to the scratch directory, never into the runtime.
RUNTIME="$("$PYTHON" -c 'from llmforge import paths; print(paths.DEVICE_RUNTIME)')"
WORK="$("$PYTHON" -c 'from llmforge import paths; print(paths.SCRATCH / "device" / "android")')"
mkdir -p "$WORK"

NDK="${NDK:-${ANDROID_NDK_HOME:-}}"
if [ -z "$NDK" ]; then
  for candidate in "$HOME/Library/Android/sdk/ndk"/*; do
    [ -d "$candidate" ] && NDK="$candidate"
  done
fi
if [ -z "$NDK" ] || [ ! -d "$NDK/toolchains/llvm/prebuilt" ]; then
  echo "Error: Android NDK not found. Set NDK or ANDROID_NDK_HOME." >&2
  exit 1
fi
HOST_TAG="darwin-x86_64"
[ -d "$NDK/toolchains/llvm/prebuilt/$HOST_TAG" ] || HOST_TAG="linux-x86_64"
ADB="${ADB:-$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")}"

# Usage: profile_tinystories_power.bash [SERIAL] [STEPS] [PROMPT]
SERIAL="${1:-${ANDROID_SERIAL:-}}"
STEPS="${2:-64}"
PROMPT="${3:-Once upon a time, a little girl named Lily}"
if [ -z "$SERIAL" ]; then
  echo "Error: pass the device serial as the first argument or set ANDROID_SERIAL." >&2
  exit 1
fi
ADB_CMD=("$ADB" "-s" "$SERIAL")

MODEL_DIR="$WORK/models/tinystories_15M"
MODEL_PT="$MODEL_DIR/stories15M.pt"
MODEL_Q8="$MODEL_DIR/stories15M.q8.bin"
TOKENIZER="$RUNTIME/tokenizer.bin"
if [ ! -f "$TOKENIZER" ] && [ -f "$MODEL_DIR/tokenizer.bin" ]; then
  TOKENIZER="$MODEL_DIR/tokenizer.bin"
fi
MODEL_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt"

mkdir -p "$MODEL_DIR"

# 1. Download the PyTorch checkpoint if not present
if [ ! -f "$MODEL_PT" ]; then
  echo "Downloading pre-trained TinyStories-15M PyTorch checkpoint from Hugging Face..."
  curl -L -o "$MODEL_PT" "$MODEL_URL"
fi

# 2. Export and quantize to INT8 Q8_0 format (.bin) if missing
EXPORT_SCRIPT="$RUNTIME/pytorch/export.py"
if [ ! -f "$EXPORT_SCRIPT" ]; then
  EXPORT_SCRIPT="$RUNTIME/export.py"
fi
if [ ! -f "$MODEL_Q8" ]; then
  echo "Quantizing TinyStories-15M to INT8 Q8_0 format (version 2)..."
  "$PYTHON" "$EXPORT_SCRIPT" "$MODEL_Q8" --checkpoint "$MODEL_PT" --version 2
fi

# 3. Detect the device architecture (32-bit armv7a / armv8l vs 64-bit aarch64)
ARCH=$("${ADB_CMD[@]}" shell "uname -m" 2>/dev/null | tr -d '\r\n' || echo "aarch64")
if [[ "$ARCH" == "armv7"* ]] || [[ "$ARCH" == "armv8l" ]]; then
  echo "Target device architecture: 32-bit ARM ($ARCH) [Pixel Watch]"
  COMPILER="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/armv7a-linux-androideabi24-clang"
else
  echo "Target device architecture: 64-bit ARM64 ($ARCH) [Phone/Emulator]"
  COMPILER="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android24-clang"
fi

# 4. Compile the runq engine for the device
RUNQ_SRC="$RUNTIME/src/runq.c"
ENGINE="$WORK/runq_tinystories_android"
echo "Compiling runq for $ARCH from $RUNQ_SRC..."
"$COMPILER" -O3 -I"$RUNTIME/src" -o "$ENGINE" "$RUNQ_SRC" -lm

# 5. Push the engine and model files to the device
echo "Pushing binaries and model files to target device..."
"${ADB_CMD[@]}" push "$ENGINE" /data/local/tmp/runq_tinystories_android
"${ADB_CMD[@]}" push "$TOKENIZER" /data/local/tmp/
"${ADB_CMD[@]}" push "$MODEL_Q8" /data/local/tmp/
"${ADB_CMD[@]}" shell "chmod +x /data/local/tmp/runq_tinystories_android"

# 6. Profile real-time power draw and energy consumption
echo "Running Real-Time Battery Power & Throughput Profiling ($STEPS steps)..."
"$PYTHON" "$SCRIPT_DIR/profile_power.py" \
  --engine runq_tinystories_android \
  --model "$(basename "$MODEL_Q8")" \
  --tokenizer "$(basename "$TOKENIZER")" \
  --tok-flag "-z" \
  --prompt "$PROMPT" \
  --steps "$STEPS" \
  --adb "$ADB" \
  --serial "$SERIAL"
