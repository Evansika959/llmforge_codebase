"""Read-only AL progress every ten minutes; independent macOS stop alarm."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def assess(record, expected_pid, runner_alive, supervisor_alive, connected=True):
    if record['pid'] != expected_pid:
        return 'Runner identity changed.'
    if record.get('ended_utc'):
        return 'Active learning stopped; exit code: ' + str(record.get('returncode'))
    if not runner_alive:
        return 'Active learning runner exited.'
    if not supervisor_alive:
        return 'Accuracy supervisor exited; inspect whether hardware is still running.'
    if not connected:
        return 'Watch disconnected from ADB.'
    return None


def sound():
    errors = []
    for _ in range(3):
        try:
            subprocess.run(['/usr/bin/say', 'Attention. Your watch active learning experiment has stopped or needs attention. Please check the experiment log.'],
                           check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(str(error))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-record', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=600)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be positive')
    workspace = args.workspace.resolve(strict=True)
    record_path = args.run_record.resolve(strict=True)
    folder = workspace/'monitoring'
    if record_path.parent != folder:
        parser.error('run record must belong to this workspace')
    record = json.loads(record_path.read_text())
    expected_pid = record['pid']
    command = record['command']
    serial = command[command.index('--serial')+1]
    with (folder/'voice_monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder/'voice_monitor.jsonl').open('a', buffering=1) as log:
            def emit(report):
                report.update(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              monitor_pid=os.getpid(), runner_pid=expected_pid)
                line = json.dumps(report)
                log.write(line+'\n')
                print(line, flush=True)
            next_progress = 0
            while True:
                try:
                    record = json.loads(record_path.read_text())
                    reason = assess(record, expected_pid, alive(expected_pid), alive(record['supervisor_pid']))
                    if not reason and time.monotonic() >= next_progress:
                        # Host transport query only; no inference temperature/power polling.
                        connection = subprocess.run(['/opt/homebrew/bin/adb', '-s', serial, 'get-state'],
                                                    capture_output=True, text=True, timeout=15)
                        connected = connection.returncode == 0 and connection.stdout.strip() == 'device'
                        reason = assess(record, expected_pid, True, True, connected)
                        state = json.loads((workspace/'state.json').read_text())
                        history = json.loads((folder/'accuracy_history.json').read_text())
                        latest = history['history'][-1]
                        emit(dict(event='progress', connected=connected, state=state,
                                  training_rows=history['training_rows'],
                                  accuracy_round=latest['round'], validation=latest['validation']))
                        next_progress = time.monotonic()+args.interval
                    if reason:
                        emit(dict(event='alert', reason=reason))
                        emit(dict(event='audio_attempt', errors=sound()))
                        return
                except Exception as error:
                    emit(dict(event='monitor_error', reason=str(error)))
                    emit(dict(event='audio_attempt', errors=sound()))
                    raise
                time.sleep(min(30, args.interval))


if __name__ == '__main__':
    main()
