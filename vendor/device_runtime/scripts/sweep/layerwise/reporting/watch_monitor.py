"""Host-only ten-minute checks and a finite audible alarm for one runner PID."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def assess(state, expected_pid, process_alive, device_connected):
    if state.get('pid') != expected_pid:
        return 'alert', 'Runner identity changed; this monitor will not follow a different run.'
    if state.get('status') == 'complete':
        return 'complete', 'All scheduled hardware measurements completed.'
    if state.get('status') == 'paused':
        return 'alert', state.get('message') or 'Experiment paused.'
    if not process_alive:
        return 'alert', 'Runner process exited without a terminal state.'
    if not device_connected:
        return 'alert', 'Watch ADB connection is unavailable.'
    return 'ok', 'Experiment running.'


def sound(message):
    # Do not unmute or change the user's volume/output device.
    errors = []
    for _ in range(3):
        try:
            subprocess.run(['/usr/bin/afplay', '/System/Library/Sounds/Glass.aiff'],
                           check=True, timeout=10, capture_output=True)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(str(error))
    try:
        subprocess.run(['/usr/bin/say', message], check=True, timeout=30,
                       capture_output=True)
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(str(error))
    return errors


def snapshot(folder, expected_pid, adb):
    state = json.loads((folder / 'run_state.json').read_text())
    connected = False
    connection_error = None
    if state.get('status') not in ('paused', 'complete'):
        try:
            # Host transport state only: no shell, temperature or power queries.
            result = subprocess.run([adb, '-s', state['serial'], 'get-state'],
                                    capture_output=True, text=True, timeout=15)
            connected = result.returncode == 0 and result.stdout.strip() == 'device'
            if not connected:
                connection_error = result.stderr.strip()
        except (OSError, subprocess.SubprocessError) as error:
            connection_error = str(error)
    kind, message = assess(state, expected_pid, alive(expected_pid), connected)
    with sqlite3.connect((folder / 'candidates.sqlite').as_uri() + '?mode=ro',
                         uri=True, timeout=10) as con:
        counts = dict(con.execute('SELECT status, count(*) FROM jobs GROUP BY status'))
    return dict(kind=kind, message=message, jobs=counts, runtime=state,
                connection_error=connection_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--runner-pid', type=int, required=True)
    parser.add_argument('--interval', type=int, default=600)
    args = parser.parse_args()
    if args.interval < 1 or args.runner_pid < 1:
        parser.error('interval and runner-pid must be positive')
    folder = args.campaign.resolve(strict=True)
    adb = shutil.which('adb')
    if not adb:
        parser.error('adb not found')
    # Separate lock: never acquire/change the hardware runner's lock.
    with (folder / 'sound_monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A sound monitor is already active for this campaign')
        with (folder / 'sound_monitor.jsonl').open('a', buffering=1) as log:
            while True:
                started = time.monotonic()
                try:
                    report = snapshot(folder, args.runner_pid, adb)
                except Exception as error:
                    report = dict(kind='alert', message='Monitor check failed: ' + str(error))
                report.update(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              monitor_pid=os.getpid())
                log.write(json.dumps(report) + '\n')
                print(json.dumps(report), flush=True)
                if report['kind'] != 'ok':
                    message = ('Watch experiment completed.' if report['kind'] == 'complete'
                               else 'Attention. Watch experiment needs your attention. Please check the experiment log.')
                    errors = sound(message)
                    log.write(json.dumps(dict(event='audio_attempt', errors=errors,
                                              monitor_pid=os.getpid())) + '\n')
                    return 2 if report['kind'] == 'alert' else 0
                remaining = args.interval - (time.monotonic() - started)
                while remaining > 0:
                    delay = min(60, remaining)
                    time.sleep(delay)
                    remaining -= delay


if __name__ == '__main__':
    raise SystemExit(main())
