"""Strict sampled admission guard; never installed inside inference sampling."""
import math
import time


def wait_ready(legacy, adb_base, target_cpu_temp=45., max_wait_sec=180.,
               poll_interval=2., min_voltage_v=3.5, min_battery_percent=30):
    if not 0 < target_cpu_temp <= 45:
        raise ValueError('Strict admission ceiling must be within (0, 45] C')
    invalid_since = None
    last_log = -float('inf')
    started = time.monotonic()
    while True:
        battery = legacy.check_battery_or_stop(adb_base,min_battery_percent)
        telem = legacy.get_device_telemetry(adb_base)
        now = time.monotonic()
        fields = ['cpu_temp_c','cooling_state','max_freq_mhz','nominal_max_freq_mhz','voltage_v']
        valid = telem and all(isinstance(telem.get(k),(int,float)) and math.isfinite(telem[k]) for k in fields)
        valid = valid and telem['cpu_temp_c']>0 and telem['voltage_v']>0
        if not valid:
            if invalid_since is None:
                invalid_since = now
            if now-invalid_since >= max_wait_sec:
                raise SystemExit('Sweep stopped: valid temperature/voltage telemetry unavailable; results retained.')
            reasons = 'telemetry unavailable'
        else:
            invalid_since = None
            reasons = []
            if telem['cpu_temp_c'] >= target_cpu_temp:
                reasons.append(f"CPU {telem['cpu_temp_c']:.1f}C >= {target_cpu_temp:g}C")
            if telem['cooling_state'] > 0:
                reasons.append(f"cooling state {telem['cooling_state']}")
            if telem['nominal_max_freq_mhz']>0 and telem['max_freq_mhz']<.975*telem['nominal_max_freq_mhz']:
                reasons.append('frequency ceiling clamped')
            if telem['voltage_v']<min_voltage_v:
                reasons.append(f"voltage {telem['voltage_v']:.2f}V")
            if not reasons:
                print(f"  [Admission] Ready: CPU {telem['cpu_temp_c']:.1f}C < {target_cpu_temp:g}C, battery {battery:g}%",flush=True)
                # Return exactly the qualifying sample, with no unvalidated reread.
                return telem
            reasons = ', '.join(reasons)
        if now-last_log >= 30:
            print(f'  [Admission] Waiting: {reasons}; battery {battery:g}%; elapsed {now-started:.0f}s. Will resume automatically.',flush=True)
            last_log = now
        # Valid but hot/throttled samples do not terminate after 180 s. Battery
        # monitoring and Ctrl-C remain active; invalid telemetry still times out.
        time.sleep(poll_interval)
