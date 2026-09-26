"""Stream one synthetic layer at a time to the existing version-2 RLM format.

This is a shape/performance fixture, NOT trained language-model weights. No ADB.
"""
import json
import math
from pathlib import Path
import struct

from .candidates import fingerprint, architecture_summary, effective_group


def validate_supported_profile(a):
    expected = dict(attention='infinite', use_concat_heads=True, mlp_variant='swiglu',
                    activation='gelu_erf', norm='rmsnorm_no_epsilon', pre_ln=True,
                    peri_ln=True, post_ln=False, bias=False, tied_embeddings=True,
                    quantization='Q8_0_global_group_backoff_from_64')
    p = a['operator_profile']
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Operator profile is not supported by this mock exporter/kernel')
    if a['n_layer'] <= 0 or len(a['layers']) != a['n_layer'] or a['d_model'] <= 0:
        raise ValueError('Invalid skeleton')
    if p['vocab_size'] <= 0 or not 1 <= p['seq_len'] <= 256 or p['rope_theta'] <= 0:
        raise ValueError('Invalid vocabulary/context/RoPE setting')
    for r in a['layers']:
        if any(r[k] <= 0 for k in ['n_h','n_kv','d_qk','d_v','d_mlp']) or r['n_h'] % r['n_kv'] or r['d_qk'] % 2:
            raise ValueError('Invalid layer geometry')
    if effective_group(a['d_model'], a['layers']) != a['q8_group_size']:
        raise ValueError('Invalid global group size')


def export_mock(architecture, output):
    validate_supported_profile(architecture)
    import torch
    from llmforge_bridge.export_llmforge_hetero import MAGIC, HEADER_BYTES, serialize_fp32, write_q8
    output=Path(output)
    if output.exists() or output.with_suffix(output.suffix+'.json').exists():
        raise ValueError('Refusing to overwrite an export or its metadata')
    output.parent.mkdir(parents=True,exist_ok=True)
    a=architecture; p=a['operator_profile']; dim=a['d_model']; group=a['q8_group_size']
    seed=int(fingerprint(a)[:16],16) % (2**63-1)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    error=[0.]
    def matrix(stream, rows, cols):
        # One matrix at a time, avoiding a full float32 360M checkpoint in RAM.
        weights=torch.randn((rows,cols),generator=generator,dtype=torch.float32)/math.sqrt(cols)
        write_q8(stream,weights,group,error)
    with output.open('xb') as stream:
        stream.write(struct.pack('<Iiiiiiiifi',MAGIC,2,a['n_layer'],dim,p['vocab_size'],p['seq_len'],1,0,p['rope_theta'],group))
        stream.write(b'\0'*(HEADER_BYTES-stream.tell()))
        for r in a['layers']:
            stream.write(struct.pack('<8i',0,r['n_h'],r['n_kv'],r['d_qk'],r['d_v'],r['d_mlp'],0,0))
        for r in a['layers']:
            for _ in range(2): serialize_fp32(stream,torch.ones(dim))
            for rows,cols in [(r['n_h']*r['d_qk'],dim),(r['n_kv']*r['d_qk'],dim),
                              (r['n_kv']*r['d_v'],dim),(dim,r['n_h']*r['d_v'])]:
                matrix(stream,rows,cols)
            for _ in range(2): serialize_fp32(stream,torch.ones(dim))
            for rows,cols in [(r['d_mlp'],dim),(r['d_mlp'],dim),(dim,r['d_mlp'])]:
                matrix(stream,rows,cols)
        serialize_fp32(stream,torch.ones(dim))
        matrix(stream,p['vocab_size'],dim)
    validate_export(a,output)
    metadata=dict(architecture_sha256=fingerprint(a),seed=seed,torch_version=torch.__version__,group_size=group,
                  synthetic_weights=True,weight_policy='torch.randn/sqrt(input_width), unit RMSNorm gains; tied embeddings; v1',
                  max_quantization_error=error[0],file_bytes=output.stat().st_size)
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def validate_export(a,path):
    from .candidates import effective_group
    if len(a['layers'])!=a['n_layer'] or effective_group(a['d_model'],a['layers'])!=a['q8_group_size']:
        raise ValueError('Architecture has inconsistent layers/group')
    with Path(path).open('rb') as stream:
        values=struct.unpack('<Iiiiiiiifi',stream.read(40))
        p=a['operator_profile']
        expected=(0x726c6d31,2,a['n_layer'],a['d_model'],p['vocab_size'],p['seq_len'],1,0,p['rope_theta'],a['q8_group_size'])
        if values!=expected: raise ValueError('Export header differs from candidate')
        stream.seek(256)
        for r in a['layers']:
            expected=(0,r['n_h'],r['n_kv'],r['d_qk'],r['d_v'],r['d_mlp'],0,0)
            if struct.unpack('<8i',stream.read(32))!=expected: raise ValueError('Export layer descriptor mismatch')
    if Path(path).stat().st_size!=architecture_summary(a)['q8_file_bytes']:
        raise ValueError('Export size/layout mismatch')
