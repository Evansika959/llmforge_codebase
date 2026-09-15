#!/usr/bin/env bash
set -euo pipefail

# Measure the random 50M to 150M Pixel Watch sweep with the historical 40 C admission settings.
#
# Environment:
#   ANDROID_SERIAL           ADB serial or IP:port of the device to measure (required)
#   ADB                      Path to adb
#   NDK                      Path to an Android NDK installation
#   PYTHON                   Python interpreter that can import llmforge (default: python)
#   LLMFORGE_DEVICE_RUNTIME  Inference runtime checkout (default: llmforge.paths.DEVICE_RUNTIME)
#   SWEEP_CONFIG             Input CSV path
#   SWEEP_OUTPUT             Result CSV path (default: under runs/device/sweeps)
#   REGENERATE_CONFIG        Set to 1 to regenerate the default CSV with seed 123

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

SWEEP_CONFIG="${SWEEP_CONFIG:-$SCRIPT_DIR/configs/watch5_random_1000_50M_150M_sweep_batch2.csv}"
if [[ -z "${SWEEP_OUTPUT:-}" ]]; then
    SWEEP_OUTPUT="$("$PYTHON" -c 'from llmforge.hw.device import runtime; print(runtime.SWEEP_OUTPUTS / "watch5_random_50M_150M_40C_decode32_batch2_results.csv")')"
fi

if [[ "${REGENERATE_CONFIG:-0}" == "1" || ! -f "$SWEEP_CONFIG" ]]; then
    "$PYTHON" "$SCRIPT_DIR/generate_watch5_random_50m_150m.py" \
        --num-samples 1000 \
        --min-params-m 50 \
        --max-params-m 150 \
        --seed 123 \
        --output "$SWEEP_CONFIG"
fi

if [[ -n "${ADB:-}" ]]; then
    ADB_COMMAND="$ADB"
elif command -v adb >/dev/null 2>&1; then
    ADB_COMMAND="$(command -v adb)"
elif [[ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]]; then
    ADB_COMMAND="$HOME/Library/Android/sdk/platform-tools/adb"
else
    echo "Error: adb was not found. Install Android SDK Platform-Tools or set ADB." >&2
    exit 1
fi

if [[ -z "${NDK:-}" ]]; then
    if [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
        NDK="$ANDROID_NDK_HOME"
    else
        for candidate in "$HOME/Library/Android/sdk/ndk"/*; do
            [[ -d "$candidate" ]] && NDK="$candidate"
        done
    fi
fi
if [[ -z "${NDK:-}" || ! -d "$NDK/toolchains/llvm/prebuilt" ]]; then
    echo "Error: Android NDK was not found. Install it or set NDK/ANDROID_NDK_HOME." >&2
    exit 1
fi
export NDK

WATCH_SERIAL="${ANDROID_SERIAL:-}"
if [[ -z "$WATCH_SERIAL" ]]; then
    echo "Error: set ANDROID_SERIAL to the serial or IP:port of the device to measure." >&2
    echo "Connected devices:" >&2
    "$ADB_COMMAND" devices >&2 || true
    exit 1
fi

ADB_TARGET=("$ADB_COMMAND" -s "$WATCH_SERIAL")
"${ADB_TARGET[@]}" get-state >/dev/null

WATCH_ARCH="$("${ADB_TARGET[@]}" shell uname -m | tr -d '\r')"
case "$WATCH_ARCH" in
    aarch64|arm64|armv7*|armv8l) ;;
    *)
        echo "Error: unsupported watch architecture '$WATCH_ARCH'." >&2
        exit 1
        ;;
esac

POWER_NODE="$("${ADB_TARGET[@]}" shell '
    if [ -r /sys/class/power_supply/battery/current_now ] && [ -r /sys/class/power_supply/battery/voltage_now ]; then
        echo /sys/class/power_supply/battery
    elif [ -r /sys/class/power_supply/sw5100_bms/current_now ] && [ -r /sys/class/power_supply/sw5100_bms/voltage_now ]; then
        echo /sys/class/power_supply/sw5100_bms
    fi
' | tr -d '\r')"
if [[ -z "$POWER_NODE" ]]; then
    echo "Error: the ADB shell cannot read a supported battery current/voltage node." >&2
    exit 1
fi

echo "Starting Pixel Watch sweep"
echo "  Arch:    $WATCH_ARCH"
echo "  Config:  $SWEEP_CONFIG"
echo "  Results: $SWEEP_OUTPUT"
echo "  Power:   $POWER_NODE"

"$PYTHON" -m llmforge.hw.device.measurement.sweep.run_sweep_configs \
    --config "$SWEEP_CONFIG" \
    --out "$SWEEP_OUTPUT" \
    --serial "$WATCH_SERIAL" \
    --adb "$ADB_COMMAND" \
    --cool-down-temp 40 \
    --steps 32 \
    --min-battery-percent 30 \
    "$@"
