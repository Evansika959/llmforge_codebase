#!/usr/bin/env python3
"""Sweep LLM architecture configurations on a physical Android device.

Run as a module:

    python -m llmforge.hw.device.measurement.sweep.run_sweep_configs \
        --config CONFIGS.csv --out RESULTS.csv --serial SERIAL

The inference runtime, meaning the kernel and power sampler sources, the model exporter and the
tokenizer, is read from llmforge.paths.DEVICE_RUNTIME, which LLMFORGE_DEVICE_RUNTIME overrides.
Build products, exported models and raw traces are temporary files in --work-dir. The process never
changes its working directory.
"""
import os
import sys
import csv
import json
import time
import shutil
import argparse
import subprocess
import numpy as np
import torch

from llmforge.hw.device import runtime as device_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = str(device_runtime.RUNTIME)
WORK_DIR = str(device_runtime.WORK)

DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "watch5_random_1000_50M_150M_sweep_batch2.csv")
DEFAULT_OUT_CSV = str(device_runtime.SWEEP_OUTPUTS / "sweep_results.csv")
LOCAL_CKPT = "temp_device_ckpt.pt"
LOCAL_RLM = "temp_device_model.q8.rlm"


def runtime_path(*parts):
    return os.path.join(RUNTIME_DIR, *parts)


def work_path(name):
    return os.path.join(WORK_DIR, name)


def find_adb():
    if os.environ.get("ADB"):
        return os.environ.get("ADB")
    which_adb = shutil.which("adb")
    if which_adb:
        return which_adb
    home_sdk_adb = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
    if os.path.isfile(home_sdk_adb):
        return home_sdk_adb
    for env_var in ["ANDROID_HOME", "ANDROID_SDK_ROOT"]:
        sdk = os.environ.get(env_var)
        if sdk and os.path.isfile(os.path.join(sdk, "platform-tools", "adb")):
            return os.path.join(sdk, "platform-tools", "adb")
    return "adb"

def find_ndk_compiler(arch):
    candidates = []
    if os.environ.get("NDK"):
        candidates.append(os.environ.get("NDK"))
    if os.environ.get("ANDROID_NDK_HOME"):
        candidates.append(os.environ.get("ANDROID_NDK_HOME"))
    
    home_ndk_dir = os.path.expanduser("~/Library/Android/sdk/ndk")
    if os.path.isdir(home_ndk_dir):
        import glob
        versions = sorted(glob.glob(os.path.join(home_ndk_dir, "*")), reverse=True)
        candidates.extend(versions)

    for ndk_root in candidates:
        bin_dir = os.path.join(ndk_root, "toolchains", "llvm", "prebuilt", "darwin-x86_64", "bin")
        if not os.path.isdir(bin_dir):
            bin_dir = os.path.join(ndk_root, "toolchains", "llvm", "prebuilt", "linux-x86_64", "bin")
        if os.path.isdir(bin_dir):
            if "armv7" in arch or "armv8l" in arch or "32" in arch:
                comp = os.path.join(bin_dir, "armv7a-linux-androideabi24-clang")
            else:
                comp = os.path.join(bin_dir, "aarch64-linux-android24-clang")
            if os.path.isfile(comp):
                return comp
    return None

def find_tokenizer_gpt2():
    for c in device_runtime.TOKENIZER_CANDIDATES:
        if os.path.exists(runtime_path(c)):
            return runtime_path(c)
    return None

def wake_and_keep_awake(adb_base):
    try:
        subprocess.run(adb_base + ["shell", "input keyevent KEYCODE_WAKEUP 2>/dev/null || true; svc power stayon true 2>/dev/null || true; settings put system screen_off_timeout 86400000 2>/dev/null || true"], capture_output=True, timeout=5)
    except Exception:
        pass

def adb_push_with_retry(adb_base, local_path, remote_path, max_retries=4):
    for attempt in range(1, max_retries + 1):
        res = subprocess.run(adb_base + ["push", local_path, remote_path], capture_output=True, text=True)
        if res.returncode == 0:
            return True
        print(f"⚠️ ADB push '{local_path}' -> '{remote_path}' failed (attempt {attempt}/{max_retries}): {res.stderr.strip()}")
        time.sleep(1.5)
        # Try reconnecting if wireless ADB target
        if "-s" in adb_base:
            s_idx = adb_base.index("-s") + 1
            serial = adb_base[s_idx]
            if ":" in serial:
                print(f"Reconnecting ADB to {serial}...")
                subprocess.run([adb_base[0], "disconnect", serial], capture_output=True)
                time.sleep(0.5)
                subprocess.run([adb_base[0], "connect", serial], capture_output=True)
                wake_and_keep_awake(adb_base)
                time.sleep(1.0)
    res.check_returncode()

def ensure_compiled_and_pushed(adb_base):
    wake_and_keep_awake(adb_base)
    arch = None
    for attempt in range(3):
        try:
            arch_res = subprocess.run(adb_base + ["shell", "uname -m"], capture_output=True, text=True, check=True)
            arch = arch_res.stdout.strip()
            if arch:
                break
        except Exception:
            if "-s" in adb_base:
                s_idx = adb_base.index("-s") + 1
                serial = adb_base[s_idx]
                if ":" in serial:
                    print(f"Connecting/Reconnecting ADB to {serial} (attempt {attempt+1}/3)...")
                    subprocess.run([adb_base[0], "disconnect", serial], capture_output=True)
                    time.sleep(0.5)
                    subprocess.run([adb_base[0], "connect", serial], capture_output=True)
                    wake_and_keep_awake(adb_base)
                    time.sleep(1.0)
            else:
                time.sleep(1.0)

    if not arch:
        print("\n" + "="*60)
        print("❌ Error: No Android device connected over ADB.")
        print("Please connect your device via USB or Wi-Fi ADB:")
        print("  adb connect <device_ip>:5555")
        print("  adb devices")
        print("="*60 + "\n")
        sys.exit(1)

    compiler = find_ndk_compiler(arch)
    if not compiler:
        print(f"Error: NDK compiler not found for {arch}. Please set NDK or ANDROID_NDK_HOME environment variable.")
        sys.exit(1)

    is_arm32 = "armv7" in arch or "armv8l" in arch or "32" in arch
    arch_flags = ["-march=armv7-a", "-mfpu=neon"] if is_arm32 else ["-march=armv8-a"]
    print(f"Compiling runq_reallm for {arch} using {os.path.basename(compiler)} (OpenMP 4-core Batched GEMM)...")
    cmd = [
        compiler,
        "-O3",
        *arch_flags,
        "-ffast-math",
        "-fopenmp",
        "-static-openmp",
        "-I" + runtime_path("src"),
        "-o",
        work_path("runq_reallm_device"),
        runtime_path("src", "runq_reallm.c"),
        "-lm",
    ]
    subprocess.run(cmd, check=True)
    
    power_sampler_source = runtime_path("src", "power_sampler.c")
    if not os.path.exists(power_sampler_source):
        raise FileNotFoundError(f"Required device sampler is missing: {power_sampler_source}")
    print("Compiling power_sampler for device...")
    cmd_ps = [compiler, "-O3", "-o", work_path("power_sampler_device"), power_sampler_source]
    subprocess.run(cmd_ps, check=True)
    adb_push_with_retry(adb_base, work_path("power_sampler_device"), "/data/local/tmp/power_sampler")
    subprocess.run(adb_base + ["shell", "chmod +x /data/local/tmp/power_sampler"], capture_output=True, check=True)
    if os.path.exists(work_path("power_sampler_device")):
        os.remove(work_path("power_sampler_device"))

    print("Pushing runq_reallm engine and tokenizer to device...")
    adb_push_with_retry(adb_base, work_path("runq_reallm_device"), "/data/local/tmp/runq_reallm")
    subprocess.run(adb_base + ["shell", "chmod +x /data/local/tmp/runq_reallm"], capture_output=True, check=True)
    if os.path.exists(work_path("runq_reallm_device")):
        os.remove(work_path("runq_reallm_device"))

    tok_path = find_tokenizer_gpt2()
    if tok_path:
        print(f"Pushing tokenizer ({tok_path}) -> /data/local/tmp/tokenizer_gpt2.bin...")
        adb_push_with_retry(adb_base, tok_path, "/data/local/tmp/tokenizer_gpt2.bin")
    else:
        print("Warning: tokenizer_gpt2.bin not found locally! Ensure it exists on device.")

    return arch

def generate_mock_ckpt(n_head, n_kv, qk, vd, mlp_hidden, n_layer, n_embd):
    vocab_size = 50257
    block_size = 256
    
    ma = {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_head_layerlist": [n_head] * n_layer,
        "n_kv_group_layerlist": [n_kv] * n_layer,
        "attention_variant_layerlist": ["infinite"] * n_layer,
        "mlp_variant": "swiglu",
        "activation_variant": "gelu",
        "norm_variant_attn": "rmsnorm",
        "norm_variant_output": "rmsnorm",
        "use_rotary_embeddings": True,
        "bias": False,
        "use_peri_ln": True,
        "use_pre_ln": True,
        "use_concat_heads": True,
    }
    
    sd = {}
    sd["transformer.wte.weight"] = torch.randn(vocab_size, n_embd)
    sd["transformer.ln_f.gain"] = torch.randn(n_embd)
    
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        sd[p + "pre_ln_attn.gain"] = torch.randn(n_embd)
        sd[p + "peri_ln_attn.gain"] = torch.randn(n_embd)
        sd[p + "pre_ln_mlp.gain"] = torch.randn(n_embd)
        sd[p + "peri_ln_mlp.gain"] = torch.randn(n_embd)
        sd[p + "mlp.c_fc_in1.weight"] = torch.randn(mlp_hidden, n_embd)
        sd[p + "mlp.c_fc_in2.weight"] = torch.randn(mlp_hidden, n_embd)
        sd[p + "mlp.c_fc_out.weight"] = torch.randn(n_embd, mlp_hidden)
        sd[p + "attn.c_attn_q.weight"] = torch.randn(n_head * qk, n_embd)
        sd[p + "attn.c_attn_k.weight"] = torch.randn(n_kv * qk, n_embd)
        sd[p + "attn.c_attn_v.weight"] = torch.randn(n_kv * vd, n_embd)
        sd[p + "attn.c_proj.weight"] = torch.randn(n_embd, n_head * vd)
        
    torch.save({"model": sd, "model_args": ma}, work_path(LOCAL_CKPT))

def export_model():
    export_script = runtime_path("reallmforge", "export_reallm_hetero.py")
    cmd = [sys.executable, export_script, work_path(LOCAL_CKPT), work_path(LOCAL_RLM), "--version", "2"]
    # The exporter runs from the runtime root, as it did inside the runtime repository.
    subprocess.run(cmd, cwd=RUNTIME_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def build_prompt_for_tokens(target_prefill_tokens):
    base_corpus = (
        "Once upon a time in a sunny meadow near a quiet village, there lived a friendly little girl named Lily. "
        "Every afternoon, Lily loved to walk along the stone path to observe the birds singing in the tall trees. "
        "One day, she noticed a fluffy gray kitten sitting beside a wooden bench. "
        "The kitten was curious and gently rubbed against her shoes with a soft purr. "
        "Lily smiled warmly and brought a bowl of fresh milk to welcome her new little friend. "
        "From that day on, the two companions explored the forest trails together, sharing joy and laughter every day. "
        "Suddenly, they heard a strange rustling sound coming from behind the bushes, and when Lily looked closely, she saw a hidden golden box tucked under the roots of a giant tree. "
        "Curious about what was inside, Lily and the kitten decided to open the box together, and to their surprise, they found a map leading to"
    )
    words = base_corpus.split()
    target_words = max(1, int(round(target_prefill_tokens / 1.3)))
    repeated_words = (words * ((target_words // len(words)) + 2))[:target_words]
    prompt_str = " ".join(repeated_words)
    if not prompt_str.endswith("because") and not prompt_str.endswith("to"):
        prompt_str += " and then they decided to explore further because"
    return prompt_str




def run_benchmark_with_power(adb_base, prefill_tokens=128, decode_steps=64, sample_rate=10.0, pre_idle_sec=4.0, post_idle_sec=4.0, min_battery_percent=30):
    adb_push_with_retry(adb_base, work_path(LOCAL_RLM), f"/data/local/tmp/{LOCAL_RLM}")
    
    prompt_str = build_prompt_for_tokens(prefill_tokens)
    total_steps = prefill_tokens + decode_steps
    interval_ms = max(5, int((1.0 / sample_rate) * 1000))
    total_est_sec = pre_idle_sec + (total_steps / 5.0) + post_idle_sec + 60.0

    remote_script = f"""#!/bin/sh
set -e
cd /data/local/tmp

input keyevent KEYCODE_WAKEUP 2>/dev/null || true
svc power stayon true 2>/dev/null || true
settings put system screen_off_timeout 86400000 2>/dev/null || true
export OMP_NUM_THREADS=4

sleep 0.8

if [ -f /sys/class/power_supply/battery/current_now ]; then
    NODE="/sys/class/power_supply/battery"
elif [ -f /sys/class/power_supply/sw5100_bms/current_now ]; then
    NODE="/sys/class/power_supply/sw5100_bms"
else
    echo "NO_NODE"
    exit 1
fi

CSV="/data/local/tmp/trace_raw.csv"
LOG="/data/local/tmp/infer_output.log"
rm -f "$CSV" "$LOG"

./power_sampler "$NODE" {total_est_sec:.1f} {interval_ms} "$CSV" &
SAMPLER_PID=$!

sleep {pre_idle_sec}

./runq_reallm {LOCAL_RLM} -g tokenizer_gpt2.bin -i '{prompt_str}' -t 0.8 -p 0.9 -n {total_steps} > "$LOG" 2>&1 &
INFER_PID=$!
# Poll during inference as well as between configurations.
BATTERY_STOP="/data/local/tmp/sweep_battery_stop"
rm -f "$BATTERY_STOP"
(
    while kill -0 "$INFER_PID" 2>/dev/null; do
        LEVEL=$(cat /sys/class/power_supply/battery/capacity 2>/dev/null || dumpsys battery | awk '/level:/ {{level=$2}} /scale:/ {{scale=$2}} END {{if (scale > 0) print int(100*level/scale)}}')
        case "$LEVEL" in
            ''|*[!0-9]*) echo "Battery telemetry unavailable" > "$BATTERY_STOP" ;;
            *) if [ "$LEVEL" -le {min_battery_percent} ]; then echo "Battery at $LEVEL%. Please charge the watch." > "$BATTERY_STOP"; fi ;;
        esac
        if [ -f "$BATTERY_STOP" ]; then
            kill "$INFER_PID" 2>/dev/null || true
            break
        fi
        sleep 1
    done
) &
GUARD_PID=$!
STATUS=0
wait "$INFER_PID" || STATUS=$?
kill "$GUARD_PID" 2>/dev/null || true
wait "$GUARD_PID" 2>/dev/null || true
if [ -f "$BATTERY_STOP" ]; then
    cat "$BATTERY_STOP"
    kill -15 "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    exit 75
fi
if [ "$STATUS" -ne 0 ]; then
    kill -15 "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    exit "$STATUS"
fi

sleep {post_idle_sec}

kill -15 "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true

echo "DONE"
"""
    push_proc = subprocess.Popen(adb_base + ["shell", "cat > /data/local/tmp/run_trace.sh && chmod +x /data/local/tmp/run_trace.sh"],
                                 stdin=subprocess.PIPE, text=True)
    push_proc.communicate(input=remote_script)

    result = subprocess.run(adb_base + ["shell", "sh /data/local/tmp/run_trace.sh"])
    if result.returncode == 75:
        raise SystemExit("Sweep stopped by battery guard. Charge the watch, unplug it, let it cool, and rerun. The interrupted configuration will be retried.")
    result.check_returncode()

    subprocess.run(adb_base + ["pull", "/data/local/tmp/trace_raw.csv", work_path("temp_trace_raw.csv")], capture_output=True, check=True)
    subprocess.run(adb_base + ["pull", "/data/local/tmp/infer_output.log", work_path("temp_infer_log.txt")], capture_output=True, check=True)
    subprocess.run(adb_base + ["shell", f"rm -f /data/local/tmp/{LOCAL_RLM} /data/local/tmp/trace_raw.csv /data/local/tmp/infer_output.log /data/local/tmp/run_trace.sh"], capture_output=True)

    with open(work_path("temp_infer_log.txt"), "r", encoding="utf-8", errors="replace") as f:
        infer_output = f.read()

    timestamps, power_vals = [], []
    with open(work_path("temp_trace_raw.csv"), "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    t = float(row[0])
                    i_ua = float(row[1])
                    v_uv = float(row[2])
                    timestamps.append(t)
                    power_vals.append((abs(i_ua) * v_uv) / 1e12)
                except ValueError:
                    pass

    for tmp in [work_path("temp_trace_raw.csv"), work_path("temp_infer_log.txt")]:
        if os.path.exists(tmp): os.remove(tmp)

    prefill_tokens_parsed = 0
    decode_tokens_parsed = 0
    prefill_ms = 0.0
    ttft_ms = 0.0
    decode_tok_s = 0.0
    total_tok_s = 0.0

    for line in infer_output.splitlines():
        if "prefill_ms:" in line:
            parts = line.split(",")
            for p in parts:
                kv = p.strip().split(":")
                if len(kv) == 2:
                    k, v = kv[0].strip(), kv[1].strip()
                    try:
                        if k == "prefill_tokens": prefill_tokens_parsed = int(v)
                        elif k == "prefill_ms": prefill_ms = float(v)
                        elif k == "ttft_ms": ttft_ms = float(v)
                        elif k == "decode_tokens": decode_tokens_parsed = int(v)
                        elif k == "decode_tok_s": decode_tok_s = float(v)
                        elif k == "achieved tok/s": total_tok_s = float(v)
                    except ValueError:
                        pass
        elif "achieved tok/s:" in line and total_tok_s == 0.0:
            try:
                total_tok_s = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    tok_s = decode_tok_s if decode_tok_s > 0 else total_tok_s
    tpot_ms = (1000.0 / tok_s) if tok_s > 0 else 0.0

    t_infer_start = pre_idle_sec
    infer_duration = (prefill_ms / 1000.0) + (decode_tokens_parsed / tok_s) if (prefill_ms > 0 and tok_s > 0) else (timestamps[-1] - pre_idle_sec - post_idle_sec)
    t_infer_end = t_infer_start + infer_duration

    # Use median of the steady pre-idle window (excluding first 1s for stabilization)
    pre_powers = [p for t, p in zip(timestamps, power_vals) if 1.0 <= t < t_infer_start]
    active_powers = [p for t, p in zip(timestamps, power_vals) if t_infer_start <= t <= t_infer_end]

    mean_baseline_w = float(np.median(pre_powers)) if pre_powers else 0.0
    mean_active_w = float(np.mean(active_powers)) if active_powers else 0.0
    peak_active_w = float(np.max(active_powers)) if active_powers else 0.0
    dynamic_power_w = max(0.0, mean_active_w - mean_baseline_w)

    total_energy_j = mean_active_w * infer_duration
    dynamic_energy_j = dynamic_power_w * infer_duration
    dynamic_energy_per_tok_mj = (dynamic_energy_j / decode_steps) * 1000.0 if decode_steps > 0 else 0.0

    return {
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "duration_s": infer_duration,
        "baseline_power_w": mean_baseline_w,
        "active_power_w": mean_active_w,
        "peak_power_w": peak_active_w,
        "dynamic_power_w": dynamic_power_w,
        "total_energy_j": total_energy_j,
        "dynamic_energy_per_token_mj": dynamic_energy_per_tok_mj
    }

def get_existing_completed_config_ids(out_csv):
    completed = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    completed.add(row[0].strip())
    return completed

def get_device_telemetry(adb_base):
    cmd = """
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0;
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0;
    cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0;
    cat /sys/class/thermal/thermal_zone17/temp 2>/dev/null || echo 0;
    cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null || echo 0;
    cat /sys/class/thermal/cooling_device1/cur_state 2>/dev/null || cat /sys/class/thermal/cooling_device0/cur_state 2>/dev/null || echo 0;
    cat /sys/class/power_supply/battery/voltage_now 2>/dev/null || echo 0;
    cat /sys/class/power_supply/battery/current_now 2>/dev/null || echo 0;
    """
    try:
        res = subprocess.run(adb_base + ["shell", cmd], capture_output=True, text=True, timeout=10)
        lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
        if len(lines) >= 8:
            freq_khz = int(lines[0]) if lines[0].isdigit() else 0
            max_freq_khz = int(lines[1]) if lines[1].isdigit() else 0
            nominal_max_freq_khz = int(lines[2]) if lines[2].isdigit() else max_freq_khz
            cpu_temp = float(lines[3]) / 1000.0 if lines[3].isdigit() else 0.0
            skin_temp = float(lines[4]) / 1000.0 if lines[4].isdigit() else 0.0
            cdev = int(lines[5]) if lines[5].isdigit() else 0
            volt = float(lines[6]) / 1e6 if lines[6].isdigit() else 0.0
            curr = float(lines[7]) / 1000.0 if (lines[7].isdigit() or lines[7].startswith("-")) else 0.0
            return {
                "freq_khz": freq_khz,
                "freq_mhz": freq_khz / 1000.0,
                "max_freq_mhz": max_freq_khz / 1000.0,
                "nominal_max_freq_mhz": nominal_max_freq_khz / 1000.0,
                "cpu_temp_c": cpu_temp,
                "skin_temp_c": skin_temp,
                "cooling_state": cdev,
                "voltage_v": volt,
                "current_ma": curr
            }
    except Exception:
        pass
    return None

def check_battery_or_stop(adb_base, min_battery_percent=30):
    """Fail closed if charge cannot be read; exit without marking a config complete."""
    try:
        result = subprocess.run(
            adb_base + ["shell", "cat /sys/class/power_supply/battery/capacity 2>/dev/null || dumpsys battery"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        raw = result.stdout.strip()
        if raw.isdigit():
            level = float(raw)
        else:
            fields = dict(line.strip().split(':', 1) for line in raw.splitlines() if ':' in line)
            level = 100.0 * float(fields['level']) / float(fields['scale'])
        if not 0 <= level <= 100:
            raise ValueError("invalid battery percentage")
    except (subprocess.SubprocessError, ValueError, KeyError, ZeroDivisionError) as exc:
        raise SystemExit(f"Sweep stopped: cannot read battery charge ({exc}). Check the watch connection and rerun.")
    if level <= min_battery_percent:
        raise SystemExit(
            f"Sweep stopped: battery is {level:.0f}% (cutoff {min_battery_percent}%). "
            "Please charge the watch, unplug it, let it cool, and rerun the same command. "
            "Completed results are saved and will be skipped on resume."
        )
    return level


def wait_for_thermal_and_voltage_recovery(adb_base, target_cpu_temp=40.0, max_wait_sec=180.0, poll_interval=2.0, min_voltage_v=3.5, min_battery_percent=30):
    """
    Monitors device thermal, CPU frequency, and battery voltage.
    Prioritizes the device-reported maximum CPU frequency and Cooling Device Level 0.
    As long as the frequency policy is unclamped and cooling state is 0,
    inference runs immediately without unnecessary thermal wait.
    """
    t0 = time.time()
    was_throttled = False

    while True:
        check_battery_or_stop(adb_base, min_battery_percent)
        telem = get_device_telemetry(adb_base)
        if not telem or telem['cpu_temp_c'] <= 0:
            time.sleep(poll_interval)
            if time.time() - t0 > max_wait_sec:
                raise SystemExit("Sweep stopped: valid temperature telemetry unavailable. Check the watch and rerun.")
            continue

        cdev = telem["cooling_state"]
        max_freq_mhz = telem["max_freq_mhz"]
        nominal_max_freq_mhz = telem["nominal_max_freq_mhz"]
        cpu_temp = telem["cpu_temp_c"]
        volt = telem["voltage_v"]

        # Compare the policy ceiling with the hardware-reported ceiling. The
        # instantaneous clock is normally low while idle and is not evidence
        # of thermal throttling.
        is_freq_clamped = (
            nominal_max_freq_mhz > 0.0
            and max_freq_mhz < nominal_max_freq_mhz * 0.975
        )
        is_cdev_active = cdev > 0
        # Secondary safety ceiling (only if target_cpu_temp > 0)
        is_hot = (target_cpu_temp > 0.0) and (cpu_temp > target_cpu_temp)
        is_volt_sagging = volt > 0 and volt < min_voltage_v

        needs_cooldown = is_freq_clamped or is_cdev_active or is_hot or is_volt_sagging

        if not needs_cooldown:
            if was_throttled:
                time.sleep(1.0)
                telem_post = get_device_telemetry(adb_base) or telem
                print(f"\n  [Frequency Guard] ✅ Max frequency restored ({telem_post['max_freq_mhz']:.0f}MHz, CoolState: 0, CPU: {telem_post['cpu_temp_c']:.1f}°C, Volt: {telem_post['voltage_v']:.2f}V). Resuming sweep.")
            return telem

        was_throttled = True
        elapsed = time.time() - t0
        if elapsed > max_wait_sec:
            raise SystemExit(f"Sweep stopped: recovery timed out after {max_wait_sec}s at {cpu_temp:.1f}°C. Let the watch cool and rerun; completed results are saved.")

        reasons = []
        if is_freq_clamped: reasons.append(f"MaxFreq={max_freq_mhz:.0f}MHz < Nominal={nominal_max_freq_mhz:.0f}MHz")
        if is_cdev_active: reasons.append(f"CoolState={cdev}")
        if is_hot: reasons.append(f"CPU={cpu_temp:.1f}°C > {target_cpu_temp:.1f}°C")
        if is_volt_sagging: reasons.append(f"Volt={volt:.2f}V < {min_voltage_v}V")
        reason_str = ", ".join(reasons)

        print(f"  [Frequency Guard] Throttling active ({reason_str})! Waiting for max frequency recovery... ({elapsed:.0f}s elapsed)", end="\r", flush=True)
        time.sleep(poll_interval)

def measure_inter_run_idle_power(adb_base, duration_sec=3.0, sample_rate=10.0):
    interval_ms = max(5, int((1.0 / sample_rate) * 1000))
    remote_script = f"""#!/bin/sh
set -e
cd /data/local/tmp

input keyevent KEYCODE_WAKEUP 2>/dev/null || true
svc power stayon true 2>/dev/null || true

if [ -f /sys/class/power_supply/battery/current_now ]; then
    NODE="/sys/class/power_supply/battery"
elif [ -f /sys/class/power_supply/sw5100_bms/current_now ]; then
    NODE="/sys/class/power_supply/sw5100_bms"
else
    echo "NO_NODE"
    exit 1
fi

CSV="/data/local/tmp/inter_idle.csv"
rm -f "$CSV"

./power_sampler "$NODE" {duration_sec:.1f} {interval_ms} "$CSV" &
SAMPLER_PID=$!

sleep {duration_sec:.1f}

kill -15 "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
echo "DONE"
"""
    push_proc = subprocess.Popen(adb_base + ["shell", "cat > /data/local/tmp/run_inter_idle.sh && chmod +x /data/local/tmp/run_inter_idle.sh"],
                                 stdin=subprocess.PIPE, text=True)
    push_proc.communicate(input=remote_script)

    subprocess.run(adb_base + ["shell", "sh /data/local/tmp/run_inter_idle.sh"], capture_output=True, check=True)
    subprocess.run(adb_base + ["pull", "/data/local/tmp/inter_idle.csv", work_path("temp_inter_idle.csv")], capture_output=True, check=True)
    subprocess.run(adb_base + ["shell", "rm -f /data/local/tmp/inter_idle.csv /data/local/tmp/run_inter_idle.sh"], capture_output=True)

    power_vals, curr_vals, volt_vals = [], [], []
    with open(work_path("temp_inter_idle.csv"), "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    i_ua = float(row[1])
                    v_uv = float(row[2])
                    p_w = (abs(i_ua) * v_uv) / 1e12
                    power_vals.append(p_w)
                    curr_vals.append(abs(i_ua) / 1000.0)
                    volt_vals.append(v_uv / 1e6)
                except ValueError:
                    pass

    if os.path.exists(work_path("temp_inter_idle.csv")):
        os.remove(work_path("temp_inter_idle.csv"))

    if not power_vals:
        return 0.0, 0.0, 0.0

    mean_p = float(np.median(power_vals))
    mean_i = float(np.median(curr_vals))
    mean_v = float(np.median(volt_vals))
    return mean_p, mean_i, mean_v

def main():
    global WORK_DIR
    parser = argparse.ArgumentParser(description="Sweep LLM architecture configurations on physical Android device with Throttling & Thermal Recovery Protection.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to input CSV config file")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_CSV, help="Path to output CSV results file")
    parser.add_argument("--serial", type=str, default=os.environ.get("ANDROID_SERIAL", ""), help="Device serial or IP:port (required, defaults to ANDROID_SERIAL)")
    parser.add_argument("--adb", type=str, default=find_adb(), help="Path to adb binary")
    parser.add_argument("--work-dir", type=str, default=WORK_DIR, help="Directory for temporary build, model and trace files")
    parser.add_argument("--prefill-tokens", type=int, default=48, help="Target prefill prompt tokens (default: 48)")
    parser.add_argument("--steps", type=int, default=32, help="Decoding steps per config (default: 32)")
    parser.add_argument("--min-battery-percent", type=int, default=30, help="Stop and request charging at or below this battery percentage (default: 30)")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="Power sampling frequency in Hz (default: 10.0)")
    parser.add_argument("--inter-run-idle-sec", "--idle-between-runs", type=float, default=0.0, help="Optional idle battery sampling duration in seconds between config runs (default: 0.0, disabled)")
    parser.add_argument("--cool-down-temp", type=float, default=40.0, help="CPU temperature ceiling before inference (°C; default: 40.0)")
    parser.add_argument("--min-cooldown-sec", type=float, default=2.0, help="Minimum pause between runs for voltage/thermal settling (default: 2.0s)")
    parser.add_argument("--max-cool-wait", type=float, default=180.0, help="Max seconds to wait for thermal/voltage recovery (default: 180.0)")
    parser.add_argument("--no-cooldown", action="store_true", help="Disable automatic thermal cooldown and voltage recovery wait")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not 0 <= args.min_battery_percent <= 100:
        parser.error("--min-battery-percent must be between 0 and 100")

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)

    if not args.serial:
        parser.error("an explicit device is required: pass --serial or set ANDROID_SERIAL (see `adb devices`)")

    WORK_DIR = os.path.abspath(args.work_dir)
    os.makedirs(WORK_DIR, exist_ok=True)

    adb_base = [args.adb]
    if args.serial:
        adb_base += ["-s", args.serial]

    print("==========================================================")
    print("Executing LLM Architecture Sweep on Physical Device")
    print(f"Config CSV   : {args.config}")
    print(f"Output CSV   : {args.out}")
    print(f"Work dir     : {WORK_DIR}")
    print(f"Device       : {args.serial if args.serial else 'default connected device'}")
    print(f"Prefill      : ~{args.prefill_tokens} prompt tokens")
    print(f"Decode       : {args.steps} generated tokens")
    print(f"Thermal Guard: Target CPU <= {args.cool_down_temp:.1f}°C (Cooldown protection: {'OFF' if args.no_cooldown else 'ON'})")
    if args.inter_run_idle_sec > 0:
        print(f"Inter-Idle   : {args.inter_run_idle_sec:.1f}s sampling between runs")
    print("==========================================================")

    print(f"Battery Guard: stop at <= {args.min_battery_percent}% and request charging")
    check_battery_or_stop(adb_base, args.min_battery_percent)
    arch = ensure_compiled_and_pushed(adb_base)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    csv_header_v2 = [
        "config_id", "category", "n_layer", "d_model", "n_h", "n_kv", "d_qk", "d_v", "d_mlp",
        "decode_tok_s", "ttft_ms", "tpot_ms", "duration_s", "baseline_power_w", "active_power_w", "peak_power_w",
        "dynamic_power_w", "total_energy_j", "dynamic_energy_per_token_mj",
        "temp_cpu_start_c", "temp_cpu_end_c", "cooling_state", "cpu_freq_mhz", "voltage_v", "notes"
    ]
    
    is_legacy_header = False
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header_v2)
    else:
        with open(args.out, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            if "temp_cpu_start_c" not in first_line:
                is_legacy_header = True

    completed_ids = get_existing_completed_config_ids(args.out)

    configs = []
    with open(args.config, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            configs.append(row)

    total_runs = len(configs)
    print(f"Loaded {total_runs} sweep configurations from {args.config} (Already completed: {len(completed_ids)})")

    for idx, cfg in enumerate(configs, 1):
        config_id = cfg["config_id"].strip()
        category = cfg.get("category", "").strip()
        n_layer = int(cfg["n_layer"])
        d_model = int(cfg["d_model"])
        n_h = int(cfg["n_h"])
        n_kv = int(cfg["n_kv"])
        d_qk = int(cfg["d_qk"])
        d_v = int(cfg["d_v"])
        d_mlp = int(cfg["d_mlp"])
        notes = cfg.get("notes", "").strip()

        if config_id in completed_ids:
            print(f"[{idx}/{total_runs}] Skipping completed: {config_id}")
            continue

        # 1. Thermal, Frequency & Voltage Recovery Guard
        check_battery_or_stop(adb_base, args.min_battery_percent)
        if not args.no_cooldown:
            telem_start = wait_for_thermal_and_voltage_recovery(
                adb_base,
                target_cpu_temp=args.cool_down_temp,
                max_wait_sec=args.max_cool_wait,
                min_battery_percent=args.min_battery_percent,
            )
        else:
            telem_start = get_device_telemetry(adb_base)

        if args.min_cooldown_sec > 0:
            time.sleep(args.min_cooldown_sec)

        if args.inter_run_idle_sec > 0:
            p_idle, i_idle, v_idle = measure_inter_run_idle_power(adb_base, duration_sec=args.inter_run_idle_sec, sample_rate=args.sample_rate)
            print(f"  [Inter-Run Idle] Battery power: {p_idle*1000.0:.1f} mW ({i_idle:.1f} mA @ {v_idle:.2f} V)")

        temp_start_c = telem_start["cpu_temp_c"] if telem_start else 0.0
        volt_start_v = telem_start["voltage_v"] if telem_start else 0.0
        print(f"\n[{idx}/{total_runs}] Profiling Config '{config_id}' ({category}) [CPU Temp: {temp_start_c:.1f}°C, Volt: {volt_start_v:.2f}V]...")
        print(f"  Specs: n_layer={n_layer}, d_model={d_model}, n_h={n_h}, n_kv={n_kv}, d_qk={d_qk}, d_v={d_v}, d_mlp={d_mlp}")

        try:
            generate_mock_ckpt(n_h, n_kv, d_qk, d_v, d_mlp, n_layer, d_model)
            export_model()

            # Recheck after export and transfer preparation, just before inference.
            check_battery_or_stop(adb_base, args.min_battery_percent)
            if not args.no_cooldown:
                telem_start = wait_for_thermal_and_voltage_recovery(
                    adb_base, target_cpu_temp=args.cool_down_temp,
                    max_wait_sec=args.max_cool_wait,
                    min_battery_percent=args.min_battery_percent,
                )
                temp_start_c = telem_start['cpu_temp_c']
            metrics = run_benchmark_with_power(
                adb_base,
                prefill_tokens=args.prefill_tokens,
                decode_steps=args.steps,
                sample_rate=args.sample_rate,
                min_battery_percent=args.min_battery_percent,
            )

            # Sample ending telemetry
            telem_end = get_device_telemetry(adb_base)
            temp_end_c = telem_end["cpu_temp_c"] if telem_end else 0.0
            cooling_state_end = telem_end["cooling_state"] if telem_end else 0
            freq_end_mhz = telem_end["freq_mhz"] if telem_end else 0.0
            volt_end_v = telem_end["voltage_v"] if telem_end else 0.0

            with open(args.out, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if is_legacy_header:
                    writer.writerow([
                        config_id, category, n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp,
                        f"{metrics['tok_s']:.4f}", f"{metrics['ttft_ms']:.2f}", f"{metrics['tpot_ms']:.2f}", f"{metrics['duration_s']:.4f}",
                        f"{metrics['baseline_power_w']:.4f}", f"{metrics['active_power_w']:.4f}",
                        f"{metrics['peak_power_w']:.4f}", f"{metrics['dynamic_power_w']:.4f}",
                        f"{metrics['total_energy_j']:.4f}", f"{metrics['dynamic_energy_per_token_mj']:.4f}",
                        notes
                    ])
                else:
                    writer.writerow([
                        config_id, category, n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp,
                        f"{metrics['tok_s']:.4f}", f"{metrics['ttft_ms']:.2f}", f"{metrics['tpot_ms']:.2f}", f"{metrics['duration_s']:.4f}",
                        f"{metrics['baseline_power_w']:.4f}", f"{metrics['active_power_w']:.4f}",
                        f"{metrics['peak_power_w']:.4f}", f"{metrics['dynamic_power_w']:.4f}",
                        f"{metrics['total_energy_j']:.4f}", f"{metrics['dynamic_energy_per_token_mj']:.4f}",
                        f"{temp_start_c:.1f}", f"{temp_end_c:.1f}", cooling_state_end, f"{freq_end_mhz:.1f}", f"{volt_end_v:.3f}",
                        notes
                    ])

            print(f"  -> decode tok/s: {metrics['tok_s']:.2f} | TTFT: {metrics['ttft_ms']:.1f}ms | TPOT: {metrics['tpot_ms']:.2f}ms/tok | active power: {metrics['active_power_w']*1000:.1f}mW | CPU Temp: {temp_start_c:.1f}°C -> {temp_end_c:.1f}°C (CoolState: {cooling_state_end})")
            check_battery_or_stop(adb_base, args.min_battery_percent)

        except Exception as e:
            print(f"  -> Error profiling config {config_id}: {e}")
        finally:
            if os.path.exists(work_path(LOCAL_CKPT)):
                os.remove(work_path(LOCAL_CKPT))
            if os.path.exists(work_path(LOCAL_RLM)):
                os.remove(work_path(LOCAL_RLM))

    print(f"\n==========================================================")
    print(f"Sweep complete! Results saved to: {args.out}")
    print("==========================================================")

if __name__ == "__main__":
    main()
