"""Initial historical dataset adapter; comparability must be confirmed before live use."""
from ..config import ROOT
from ..data.dataset import MeasurementDataset
from ..data.legacy import load_cohort
from .storage import digest


def legacy_dataset(previous, energy='dynamic', device_id='PixelWatch5-historical-identity-unverified',temperature=40.):
    snapshot,_,rows,configs,ix,_,gross,_ = load_cohort(previous)
    validation,test = set(ix['fixed_validation']),set(ix['pooled_test'])
    observations = []
    for i,(row,c) in enumerate(zip(rows,configs)):
        observations.append(dict(measurement_id='legacy:'+row['config_id'],config_id=row['config_id'],
            round_id='legacy_snapshot',split='test' if i in test else 'validation' if i in validation else 'train',
            architecture=c,metrics=dict(decode_tok_s=float(row['decode_tok_s']),ttft_ms=float(row['ttft_ms']),
                dynamic_energy_per_token_mj=float(row['dynamic_energy_per_token_mj']),gross_energy_per_token_mj=float(gross[i]))))
    protocol = dict(protocol_id='legacy_random_49x32_active_v1',device_id=device_id,
        kernel_id=digest(ROOT/'src/runq_llmforge.c'),prompt_tokens=49,output_tokens=32,energy_target=energy,
        adapter='legacy_random_sweep_v1',temperature_ceiling=temperature,min_battery_percent=30,
        model_seed_policy='active rounds: deterministic architecture SHA256; historical measurements: unseeded',
        comparability='Historical device/build/state not fully verified. Operator confirmation required; reference measurements are optional.')
    if energy=='dynamic':
        protocol['energy_label_validation']='baseline_reconciled_v1'
    return MeasurementDataset(dict(schema_version=1,protocol=protocol,observations=observations,source_hashes=snapshot['hashes']))
