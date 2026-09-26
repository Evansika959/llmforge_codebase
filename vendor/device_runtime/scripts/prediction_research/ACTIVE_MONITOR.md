# Independent active-learning voice monitor

Run `python -m scripts.prediction_research.active_watch_monitor --workspace WORKSPACE
--run-record WORKSPACE/monitoring/run_TIMESTAMP.json --interval 600` after starting
the active-learning runner. Use the exact run record, not a historical PID.

The monitor appends progress and fixed-validation errors every ten minutes to
`monitoring/voice_monitor.jsonl`. Runner/supervisor exit is checked every 30 seconds;
host ADB transport is checked every ten minutes. It does not query watch temperature
or power during inference, retrain, restart the experiment, or alter measurements.
Its source is outside the frozen production prediction source set.

On termination, disconnection or monitor error, it speaks an English warning three
times using macOS `say`, then exits. It does not change audio volume or routing.
Keep the host awake and its audio audible. This independent process provides local
logging and audio, not scheduled chat delivery. An active assistant turn can read
the reports and relay them in chat. Stop only the monitor PID to silence monitoring
without stopping hardware. Restart it with the new run record after resuming.

Tests: `python -m unittest scripts.prediction_research.test_active_watch_monitor -v`.
