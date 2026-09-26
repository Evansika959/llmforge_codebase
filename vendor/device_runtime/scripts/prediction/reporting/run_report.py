"""Reporting consumes saved results; never trains a model or chooses candidates."""
import json
from pathlib import Path


def write_report(output, results):
    output = Path(output)
    lines = ['# Predictor evaluation', '', 'Fixed held-out architectures; no test labels used for fitting or stopping.', '',
             '| Target | MAPE | MAE | R² |', '|---|---:|---:|---:|']
    for r in results:
        r2 = 'n/a' if r['r2'] is None else f"{r['r2']:.3f}"
        lines.append(f"| {r['target']} | {r['mape']:.2f}% | {r['mae']:.3f} | {r2} |")
    with (output/'README.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    args = parser.parse_args()
    write_report(args.results, json.loads((args.results/'metrics.json').read_text()))
