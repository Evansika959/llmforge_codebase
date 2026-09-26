"""Architecture-only analytic workload proxies."""
from ..config import FEATURES

def physical_features(config, prompt=49, decode=31, vocab=50257):
    """MAC counts exclude norms/activation functions. Bytes are logical traffic proxies.

    No measured bandwidth, cache size, power or frequency is assumed. Cache reuse,
    compiler vectorization and real DRAM traffic cannot be inferred exactly here.
    Fixed workload matches the original runner: first output follows prefill, then
    31 decode forward calls, for normally 32 output tokens.
    """
    c={k:float(config[k]) for k in FEATURES[:-2]}
    L,D,H,K,Q,V,M,G=(c[k] for k in ['n_layer','d_model','n_h','n_kv','d_qk','d_v','d_mlp','q8_group_size'])
    c['kv_bytes_per_token']=4*L*K*(Q+V)
    c['attention_width']=H*Q
    attention_weights=D*(H*Q+K*Q+K*V+H*V)
    mlp_weights=3*D*M
    weights=L*(attention_weights+mlp_weights)+vocab*D
    context=prompt+(decode+1)/2
    dense_decode=weights
    attention_decode=L*H*(Q+V)*context
    prefill_dense=prompt*L*(attention_weights+mlp_weights)+vocab*D
    prefill_attn=L*H*(Q+V)*prompt*(prompt+1)/2
    compute=dict(
        decode_dense_mac=dense_decode, decode_attention_mac=attention_decode,
        decode_total_mac=dense_decode+attention_decode,
        prefill_dense_mac=prefill_dense, prefill_attention_mac=prefill_attn,
        prefill_total_mac=prefill_dense+prefill_attn,
        sequence_mac=prefill_dense+prefill_attn+decode*(dense_decode+attention_decode),
        mlp_mac_fraction=L*mlp_weights/weights,
        classifier_mac_fraction=vocab*D/weights,
        attention_projection_mac=L*attention_weights,
        decode_output_elements=L*(H*Q+K*Q+K*V+2*D+2*M)+vocab,
    )
    weight_bytes=weights*(1+4/G)
    cache_read_unique=4*L*K*(Q+V)*context
    cache_read_per_head=4*L*H*(Q+V)*context
    cache_write=4*L*K*(Q+V)
    traffic=weight_bytes+cache_read_per_head+cache_write
    memory=dict(
        q8_weight_bytes=weight_bytes, q8_scale_bytes=weights*4/G,
        embedding_fp32_bytes=vocab*D*4,
        kv_read_unique_bytes=cache_read_unique,
        kv_read_per_head_bytes=cache_read_per_head,
        kv_write_bytes=cache_write,
        kv_allocated_bytes=cache_write*256,
        decode_logical_bytes=traffic,
        decode_mac_per_logical_byte=(dense_decode+attention_decode)/traffic,
        prefill_mac_per_weight_byte=(prefill_dense+prefill_attn)/weight_bytes,
        layer_weight_bytes=(attention_weights+mlp_weights)*(1+4/G),
        prefill_mlp_buffer_bytes=prompt*M*4,
    )
    kernel=dict(
        decode_general_path_mac=weights*int(G==64),
        decode_neon16_path_mac=weights*int(G==16),
        decode_neon32_path_mac=weights*int(G==32),
        quantized_group_count=weights/G,
        omp_region_count=7*L+1,
        mac_per_omp_region=weights/(7*L+1),
        gqa_reuse_factor=H/K,
    )
    return c, compute, memory, kernel
