"""Aligned energy integration and independent performance/energy quality flags."""
import csv
import json
import math
from pathlib import Path
import statistics


def integrate(points, start, end):
    if not points or start >= end or start < points[0][0] or end > points[-1][0]:
        raise ValueError('Trace does not bracket the measurement window')
    total = 0.
    for (t0,p0),(t1,p1) in zip(points,points[1:]):
        if t1 <= t0: raise ValueError('Nonmonotonic power timestamps')
        left,right = max(t0,start),min(t1,end)
        if left >= right: continue
        if t1-t0 > .5: raise ValueError('Power trace gap exceeds 500 ms')
        pl = p0+(p1-p0)*(left-t0)/(t1-t0)
        pr = p0+(p1-p0)*(right-t0)/(t1-t0)
        total += (pl+pr)*.5*(right-left)
    return total


def parse(folder):
    folder=Path(folder)
    timing=json.loads((folder/'timing.json').read_text())
    if [timing.get(k) for k in ['prefill_tokens','output_tokens','decode_tokens']] != [49,32,31]:
        raise ValueError('Incomplete fixed workload')
    marks=[timing[k] for k in ['idle_start','start','prefill_end','end','post_end']]
    if not all(math.isfinite(t) for t in marks) or any(b<=a for a,b in zip(marks,marks[1:])):
        raise ValueError('Invalid phase timestamps')
    if not 0 < timing['admission_temperature_c'] < 45:
        raise ValueError('Invalid admission temperature')
    duration=timing['end']-timing['start']
    result=dict(decode_tok_s=31/(timing['end']-timing['prefill_end']),
                ttft_ms=1000*(timing['prefill_end']-timing['start']),duration_s=duration,
                dynamic_energy_per_token_mj=None,baseline_power_w=None,active_power_w=None,
                energy_valid=False,timing=timing)
    try:
        points=[]
        with (folder/'trace.csv').open() as stream:
            for row in csv.reader(stream):
                t,i,v=map(float,row)
                if not all(math.isfinite(x) for x in [t,i,v]) or not 2e6<v<6e6 or abs(i)>1e7:
                    raise ValueError('Invalid power sensor values')
                points.append((t,abs(i)*v/1e12))
        pre=[p for t,p in points if timing['idle_start']+1 <= t <= timing['start']-.2]
        post=[p for t,p in points if timing['end']+1 <= t <= timing['post_end']-.2]
        if min(len(pre),len(post))<15: raise ValueError('Insufficient baseline samples')
        baseline=statistics.median(pre); post_power=statistics.median(post)
        gross=integrate(points,timing['start'],timing['end'])
        decode_gross=integrate(points,timing['prefill_end'],timing['end'])
        dynamic=gross-baseline*duration
        drift=abs(post_power-baseline)/max(baseline,.01)
        result.update(baseline_power_w=baseline,post_baseline_power_w=post_power,
            active_power_w=gross/duration,baseline_relative_drift=drift,
            gross_energy_per_token_mj=1000*gross/32,raw_dynamic_energy_per_token_mj=1000*dynamic/32,
            decode_gross_energy_per_forward_mj=1000*decode_gross/31,
            energy_valid=dynamic>0,energy_warning='baseline_drift_over_25pct' if drift>.25 else None,
            dynamic_energy_per_token_mj=1000*dynamic/32 if dynamic>0 else None)
        if dynamic<=0: result['energy_warning']='nonpositive_dynamic_energy'
    except (ValueError,OSError) as error:
        result['energy_warning']=str(error)
    return result
