"""Deterministic 500-model layerwise design; standard library only, no ADB."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import random


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def effective_group(dim, layers):
    group = 64
    widths = [dim] + [w for r in layers for w in [r['n_h']*r['d_v'], r['d_mlp']]]
    while any(w % group for w in widths):
        group //= 2
    if group not in (16, 32, 64):
        raise ValueError('Unsupported group size')
    return group


def layer_shapes(family):
    def kv_choices(h):
        if 'n_kv_by_n_h' not in family:
            return [family['n_kv']]
        choices = family['n_kv_by_n_h'][str(h)]
        if not choices or len(set(choices)) != len(choices) or any(type(k) is not int or k<=0 or h%k for k in choices):
            raise ValueError('KV-head choices must be distinct positive divisors of attention heads')
        return choices
    return [dict(n_h=h, n_kv=k, d_qk=q, d_v=v, d_mlp=m)
            for q, v, h, m in itertools.product(family['d_qk'], family['d_v'], family['n_h'], family['d_mlp'])
            for k in kv_choices(h)]


def validate_architecture(arch, spec):
    family = spec['families'][arch['family']]
    if (arch['n_layer'] != family['n_layer'] or arch['d_model'] != family['d_model'] or
            arch['operator_profile'] != spec['operator_profile'] or len(arch['layers']) != arch['n_layer']):
        raise ValueError('Skeleton/operator mismatch')
    allowed = {canonical(r) for r in layer_shapes(family)}
    for layer in arch['layers']:
        if canonical(layer) not in allowed or layer['n_h'] % layer['n_kv'] or layer['d_qk'] % 2:
            raise ValueError('Invalid per-layer shape/GQA/RoPE')
    if effective_group(arch['d_model'], arch['layers']) != arch['q8_group_size']:
        raise ValueError('Declared group does not match global backoff')


def architecture_summary(arch):
    dim, group = arch['d_model'], arch['q8_group_size']
    profile = arch['operator_profile']
    layer_matrices = [dim*(r['n_h']*r['d_qk']+r['n_kv']*(r['d_qk']+r['d_v'])+r['n_h']*r['d_v'])
                      + 3*dim*r['d_mlp'] for r in arch['layers']]
    embedding = profile['vocab_size']*dim
    matrices = sum(layer_matrices)+embedding
    norms = 4*dim*arch['n_layer']+dim
    weight_bytes = matrices + (matrices//group)*4 + norms*4
    kv_bytes = sum(4*profile['seq_len']*r['n_kv']*(r['d_qk']+r['d_v']) for r in arch['layers'])
    return dict(total_params=matrices+norms, q8_file_bytes=256+32*arch['n_layer']+weight_bytes,
                dequantized_embedding_bytes=4*embedding, kv_cache_bytes=kv_bytes,
                memory_lower_bound_bytes=weight_bytes+4*embedding+kv_bytes,
                unique_layer_shapes=len({canonical(r) for r in arch['layers']}),
                layer_shape_transitions=sum(a!=b for a,b in zip(arch['layers'],arch['layers'][1:])))


def generate(spec):
    if spec['schema_version'] not in (1,2) or len(spec['families']) != 2:
        raise ValueError('Expected two-family search space v1 or v2')
    rng = random.Random(spec['seed'])
    rows, seen = [], set()
    exclusions = spec.get('exclusions', {})
    excluded_ids = set(exclusions.get('architecture_sha256', []))
    excluded_groups = set(exclusions.get('permutation_groups', []))
    require_variable_kv = spec.get('require_variable_kv_in_heterogeneous_candidates', False)
    for name, family in spec['families'].items():
        shapes = layer_shapes(family)
        if (spec['schema_version']==1 and len(shapes)!=192) or not shapes or family['n_layer'] < 4 or family['d_model'] % 64:
            raise ValueError('Invalid layer space/depth/residual width')
        depth = family['n_layer']
        for target, count in [(16,84),(32,83),(64,83)]:
            local = lambda r: effective_group(family['d_model'],[r])
            pool = [r for r in shapes if local(r) >= target]
            exact = [r for r in pool if local(r) == target]

            def add(layers, pattern):
                arch = dict(family=name,n_layer=depth,d_model=family['d_model'],
                            operator_profile=spec['operator_profile'],q8_group_size=target,layers=layers)
                validate_architecture(arch,spec)
                sha = fingerprint(arch)
                multiset = dict(arch, layers=sorted(layers,key=canonical))
                permutation_group = fingerprint(multiset)
                if sha in seen or sha in excluded_ids or permutation_group in excluded_groups:
                    return False
                if pattern!='uniform_control' and require_variable_kv and len({r['n_kv'] for r in layers})<2:
                    return False
                seen.add(sha)
                # Group permutations together for future evaluation splitting.
                rows.append(dict(candidate_id='LW_'+sha[:20],architecture_sha256=sha,architecture=arch,
                                 permutation_group=permutation_group,pattern=pattern,summary=architecture_summary(arch)))
                return True

            if spec['schema_version']==1 and not exclusions:
                for shape in rng.sample(exact,8):
                    add([shape.copy() for _ in range(depth)],'uniform_control')
            else:
                done=0
                for shape in rng.sample(exact,len(exact)):
                    done+=add([shape.copy() for _ in range(depth)],'uniform_control')
                    if done==8: break
                if done!=8: raise ValueError('Cannot fill nonoverlapping uniform controls')
            for pattern, needed in [('blockwise',16),('alternating',16),('random',count-56)]:
                done = 0
                attempts = 0
                while done < needed:
                    attempts += 1
                    if attempts > 10000:
                        raise ValueError('Cannot fill unique stratum')
                    if pattern == 'random':
                        layers = [rng.choice(pool).copy() for _ in range(depth)]
                        layers[rng.randrange(depth)] = rng.choice(exact).copy()
                    else:
                        types = rng.sample(pool,rng.randint(2,4))
                        types[0] = rng.choice(exact)
                        layers = [types[(i*len(types)//depth) if pattern=='blockwise' else i%len(types)].copy()
                                  for i in range(depth)]
                    if len({canonical(r) for r in layers}) < 2:
                        continue
                    done += add(layers,pattern)
            for _ in range(8):
                for attempt in range(10000):
                    layers = [rng.choice(pool).copy() for _ in range(depth)]
                    layers[rng.randrange(depth)] = rng.choice(exact).copy()
                    arranged = sorted(layers,key=canonical)
                    shuffled = [r.copy() for r in layers]
                    rng.shuffle(shuffled)
                    if arranged == shuffled:
                        continue
                    arches = [dict(family=name,n_layer=depth,d_model=family['d_model'],operator_profile=spec['operator_profile'],
                                   q8_group_size=target,layers=ls) for ls in [arranged,shuffled]]
                    if (require_variable_kv and len({r['n_kv'] for r in layers})<2) or any(
                            fingerprint(a) in seen or fingerprint(a) in excluded_ids or
                            fingerprint(dict(a,layers=sorted(a['layers'],key=canonical))) in excluded_groups for a in arches):
                        continue
                    add(arranged,'permutation_grouped')
                    add(shuffled,'permutation_shuffled')
                    break
                else:
                    raise ValueError('Cannot fill permutation pairs')

    # Fixed split before any performance labels exist, stratified by family/G.
    # A multiset and all its layer-order permutations stay in the same split.
    buckets = defaultdict(list)
    for row in rows:
        a=row['architecture']; buckets[(a['family'],a['q8_group_size'])].append(row)
    for (family,group), bucket in buckets.items():
        groups=defaultdict(list)
        for row in bucket: groups[row['permutation_group']].append(row)
        remaining=list(groups.values()); rng.shuffle(remaining)
        quota=9 if group==16 else 8
        for split in ['validation','test']:
            need=quota
            for pack in list(remaining):
                if len(pack)<=need:
                    for row in pack: row['split']=split
                    need-=len(pack); remaining.remove(pack)
                if need==0: break
            if need: raise ValueError('Cannot allocate grouped holdout quota')
        for pack in remaining:
            for row in pack: row['split']='train'
        rng.shuffle(bucket)
    schedule=[]
    # 50 blocks of 10, each containing 5 models from each family.
    for batch in range(50):
        block=[]
        for family in spec['families']:
            for i in range(5):
                group=[16,32,64][(batch*5+i)%3]
                block.append(buckets[(family,group)].pop())
        rng.shuffle(block)
        for row in block:
            row['ordinal']=len(schedule)+1; row['batch']=batch+1; schedule.append(row)
    if len(schedule)!=500 or any(buckets.values()):
        raise ValueError('Incorrect candidate allocation')
    return schedule
