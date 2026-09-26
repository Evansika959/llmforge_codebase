"""Measurement validation and descriptive anchor drift gates (not causal tests)."""
import math
import numpy as np

from ..config import FEATURES


def baseline_energy(row, output_tokens=32):
    """Validate the recorded baseline-subtracted label within CSV rounding error.

    Do not subtract again from an already dynamic label or clamp it to epsilon.
    Invalid/nonpositive labels must be remeasured/reviewed, not silently trained.
    """
    active, baseline, duration, dynamic = [float(row[k]) for k in
        ['active_power_w','baseline_power_w','duration_s','dynamic_energy_per_token_mj']]
    if not all(math.isfinite(v) and v>0 for v in [active,baseline,duration,dynamic]):
        raise ValueError('Missing/nonpositive/nonfinite baseline or dynamic energy; review or remeasure')
    expected = max(active-baseline,0)*duration*1000/output_tokens
    tolerance = (.0001*duration+.00005*abs(active-baseline))*1000/output_tokens+.0001
    if abs(expected-dynamic)>tolerance:
        raise ValueError('Dynamic energy does not reconcile with baseline subtraction')
    return dynamic


def result_metrics(row, config, protocol):
    if None in row or any(v is None for v in row.values()):
        raise ValueError('Malformed/incomplete measurement row')
    if row['config_id'] != config['config_id']:
        raise ValueError('Wrong measurement identity')
    if any(float(row[k]) != float(config[k]) for k in FEATURES[:7]):
        raise ValueError('Measured architecture mismatch')
    if row.get('notes','').strip() or row.get('quality_flags','').strip() not in ('','none'):
        raise ValueError('Flagged measurement needs review')
    if row.get('output_tokens') and int(row['output_tokens']) != protocol['output_tokens']:
        raise ValueError('Output token count mismatch')
    values = {k:float(row[k]) for k in ['decode_tok_s','ttft_ms']}
    energy = protocol['energy_target']
    if energy=='gross':
        values['gross_energy_per_token_mj'] = float(row['total_energy_j'])*1000/protocol['output_tokens']
    else:
        values['dynamic_energy_per_token_mj'] = (baseline_energy(row,protocol['output_tokens'])
            if protocol.get('energy_label_validation')=='baseline_reconciled_v1'
            else float(row['dynamic_energy_per_token_mj']))
    if any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError('Nonpositive/nonfinite target')
    return values


def anchor_report(anchors, metrics_by_id, previous=None, tolerance=.25):
    if not anchors:
        return dict(passed=True,enabled=False,comparisons=[],
                    warning='Anchor gate disabled by acquisition policy; device drift was not assessed. Measurement validation remains required.')
    comparisons = []
    current = {}
    targets = list(metrics_by_id[anchors[0]['before']])
    for anchor in anchors:
        before,after = metrics_by_id[anchor['before']],metrics_by_id[anchor['after']]
        current[anchor['config_id']] = {t:float(np.sqrt(before[t]*after[t])) for t in targets}
        for t in targets:
            comparisons.append(dict(config_id=anchor['config_id'],target=t,kind='within_round',ratio=after[t]/before[t]))
            if previous and anchor['config_id'] in previous:
                comparisons.append(dict(config_id=anchor['config_id'],target=t,kind='between_rounds',
                                        ratio=current[anchor['config_id']][t]/previous[anchor['config_id']][t]))
    # Symmetric proportional threshold, plus an individual-anchor extreme gate.
    threshold = math.log1p(tolerance)
    failed = []
    for kind in {r['kind'] for r in comparisons}:
        for target in targets:
            deviations = [abs(math.log(r['ratio'])) for r in comparisons if r['kind']==kind and r['target']==target]
            if np.median(deviations) > threshold or max(deviations) > 2*threshold:
                failed.append(dict(kind=kind,target=target))
    return dict(passed=not failed,enabled=True,tolerance=tolerance,failed=failed,
                comparisons=comparisons,reference=current,
                interpretation='Heuristic gate on repeated architectures, not a confidence test or automatic denoising rule.')
