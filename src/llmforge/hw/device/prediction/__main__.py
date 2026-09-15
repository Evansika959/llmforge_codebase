"""CLI entry point. Run from the repository root: python -m llmforge.hw.device.prediction."""
import argparse
import importlib
import sys

COMMANDS = {
    "active": "llmforge.hw.device.prediction.active_learning.cli",
    "train": "llmforge.hw.device.prediction.training.run",
    "predict": "llmforge.hw.device.prediction.inference.predict_proposed_physics_surrogate",
    "dataset": "llmforge.hw.device.prediction.data.rounds",
    "report": "llmforge.hw.device.prediction.reporting.run_report",
    "compare-surrogates": "llmforge.hw.device.prediction.experiments.compare_surrogates",
    "compare-physics-priors": "llmforge.hw.device.prediction.experiments.compare_physics_priors",
    "compare-search-surrogates": "llmforge.hw.device.prediction.experiments.compare_search_surrogates",
    "train-proposed-physics-surrogate": "llmforge.hw.device.prediction.experiments.train_proposed_physics_surrogate",
    "evaluate-batch2-progress": "llmforge.hw.device.prediction.experiments.evaluate_batch2_progress",
    "evaluate-gross-energy": "llmforge.hw.device.prediction.experiments.evaluate_gross_energy",
    "transformer-learning-curve": "llmforge.hw.device.prediction.experiments.transformer_learning_curve",
    "diagnose-surrogate-error": "llmforge.hw.device.prediction.experiments.diagnose_surrogate_error",
    "investigate-state-signal": "llmforge.hw.device.prediction.experiments.investigate_state_signal",
    "audit-gross-energy": "llmforge.hw.device.prediction.evaluation.audit_gross_energy",
    "audit-proposed-physics-surrogate": "llmforge.hw.device.prediction.evaluation.audit_proposed_physics_surrogate",
    "report-batch2-progress": "llmforge.hw.device.prediction.reporting.report_batch2_progress",
    "report-surrogates": "llmforge.hw.device.prediction.reporting.report_surrogates",
    "report-error-diagnosis": "llmforge.hw.device.prediction.reporting.report_error_diagnosis",
    "plot-state-signal": "llmforge.hw.device.prediction.reporting.plot_state_signal",
    "predict-surrogates": "llmforge.hw.device.prediction.inference.predict_surrogates",
    "predict-proposed-physics-surrogate": "llmforge.hw.device.prediction.inference.predict_proposed_physics_surrogate"
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=sorted(COMMANDS))
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.print_help()
        return
    command = parser.parse_args(arguments[:1]).command
    entry = importlib.import_module(COMMANDS[command]).main
    original = sys.argv
    try:
        sys.argv = [f'python -m llmforge.hw.device.prediction {command}', *arguments[1:]]
        entry()
    finally:
        sys.argv = original


if __name__ == '__main__':
    main()
