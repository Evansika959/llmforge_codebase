"""CLI entry point. Run from the repository root: python -m scripts.prediction."""
import argparse
import importlib
import sys

COMMANDS = {
    "active": "scripts.prediction.active_learning.cli",
    "train": "scripts.prediction.training.run",
    "predict": "scripts.prediction.inference.predict_proposed_physics_surrogate",
    "dataset": "scripts.prediction.data.rounds",
    "report": "scripts.prediction.reporting.run_report",
    "compare-surrogates": "scripts.prediction.experiments.compare_surrogates",
    "compare-physics-priors": "scripts.prediction.experiments.compare_physics_priors",
    "compare-search-surrogates": "scripts.prediction.experiments.compare_search_surrogates",
    "train-proposed-physics-surrogate": "scripts.prediction.experiments.train_proposed_physics_surrogate",
    "evaluate-batch2-progress": "scripts.prediction.experiments.evaluate_batch2_progress",
    "evaluate-gross-energy": "scripts.prediction.experiments.evaluate_gross_energy",
    "transformer-learning-curve": "scripts.prediction.experiments.transformer_learning_curve",
    "diagnose-surrogate-error": "scripts.prediction.experiments.diagnose_surrogate_error",
    "investigate-state-signal": "scripts.prediction.experiments.investigate_state_signal",
    "audit-gross-energy": "scripts.prediction.evaluation.audit_gross_energy",
    "audit-proposed-physics-surrogate": "scripts.prediction.evaluation.audit_proposed_physics_surrogate",
    "report-batch2-progress": "scripts.prediction.reporting.report_batch2_progress",
    "report-surrogates": "scripts.prediction.reporting.report_surrogates",
    "report-error-diagnosis": "scripts.prediction.reporting.report_error_diagnosis",
    "plot-state-signal": "scripts.prediction.reporting.plot_state_signal",
    "predict-surrogates": "scripts.prediction.inference.predict_surrogates",
    "predict-proposed-physics-surrogate": "scripts.prediction.inference.predict_proposed_physics_surrogate"
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
        sys.argv = [f'python -m scripts.prediction {command}', *arguments[1:]]
        entry()
    finally:
        sys.argv = original


if __name__ == '__main__':
    main()
