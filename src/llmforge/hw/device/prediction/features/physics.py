"""Architecture-only physics features and train-calibrated hardware profile."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls

FEATURES32 = ['n_layer','d_model','n_h','n_kv','d_qk','d_v','d_mlp','vocab_size',
    'q8_group_size','total_params','model_weight_bytes','layer_weight_bytes',
    'attention_output_width','output_projection_params','mlp_expansion','gqa_ratio',
    'kv_bytes_per_token','cached_layer_weight_bytes','spilled_layer_weight_bytes',
    'cache_residency_fraction','layer_working_set_bytes','cache_pressure_ratio',
    'hidden_aligned64','mlp_aligned64','attention_output_aligned64','general_gs64_path',
    'decode_mac','prefill_mac','decode_roofline_s','prefill_roofline_s',
    'decode_sync_fraction','prefill_dispatch_fraction']
TOKEN_GROUPS = [[7,8,9,10],[0,28,30],[1,22],[2,3,4,5,12,13,15,16,24],
                [6,14,23],[11,17,18,19,20,21,25],[26,27,29,31]]
assert sorted(sum(TOKEN_GROUPS,[])) == list(range(32))


def architecture_stats(config,prompt_tokens=49):
    L,D,H,K,Q,V,M=[int(config[k]) for k in ['n_layer','d_model','n_h','n_kv','d_qk','d_v','d_mlp']]
    vocab=int(config.get('vocab_size',50257)); G=64
    while any(d%G for d in [D,H*V,M]):G//=2
    attention=D*(H*Q+K*Q+K*V+H*V);mlp=3*D*M
    matrices=attention+mlp; embed=vocab*D
    layer=matrices*(1+4/G)+16*D
    model=L*layer+embed*(1+4/G)+4*D
    kv=4*L*K*(Q+V)
    decode=L*matrices+embed+L*H*(Q+V)*(prompt_tokens+16)
    prefill=prompt_tokens*L*matrices+embed+L*H*(Q+V)*prompt_tokens*(prompt_tokens+1)/2
    return dict(L=L,D=D,H=H,K=K,Q=Q,V=V,M=M,vocab=vocab,G=G,
        params=L*(matrices+4*D)+embed+D,layer=layer,model=model,kv=kv,decode=decode,prefill=prefill)


@dataclass
class HardwareProfile:
    cache_bytes: float = 524288.
    prompt_tokens: int = 49
    bandwidth_bytes_s: float = 1e9
    compute_mac_s: float = 1e9
    sync_s_layer: float = 0.
    dispatch_s_layer: float = 0.
    source: str = 'training-only NNLS effective constants; shared L2 512KiB read from Pixel Watch 5 sysfs'

    def fit(self, configs, y):
        """Fit effective positive costs, not physical peak hardware specifications.
        y order: throughput tok/s, TTFT ms, prefill-inclusive energy mJ/output token.
        """
        st=[architecture_stats(c,self.prompt_tokens) for c in configs]
        A=np.array([[s['model']/1e6,s['L']] for s in st])
        B=np.array([[s['prefill']/1e9,s['L']] for s in st])
        decode,_=nnls(A,1/y[:,0]);prefill,_=nnls(B,y[:,1]/1000)
        self.bandwidth_bytes_s=1e6/max(float(decode[0]),1e-12)
        self.sync_s_layer=float(decode[1]);self.compute_mac_s=1e9/max(float(prefill[0]),1e-12)
        self.dispatch_s_layer=float(prefill[1])
        return self

    def transform(self,configs):
        result=[]
        for c in configs:
            s=architecture_stats(c,self.prompt_tokens)
            L,D,H,K,Q,V,M,G=[s[k] for k in ['L','D','H','K','Q','V','M','G']]
            cached=min(s['layer'],self.cache_bytes);spill=max(0,s['layer']-self.cache_bytes)
            working=s['layer']+4*K*(Q+V)*(self.prompt_tokens+16)
            dt=s['model']/self.bandwidth_bytes_s+L*self.sync_s_layer
            pt=s['prefill']/self.compute_mac_s+L*self.dispatch_s_layer
            result.append([L,D,H,K,Q,V,M,s['vocab'],G,s['params'],s['model'],s['layer'],
                H*V,D*H*V,M/D,K/H,s['kv'],cached,spill,cached/s['layer'],working,
                working/self.cache_bytes,int(D%64==0),int(M%64==0),int((H*V)%64==0),int(G==64),
                s['decode'],s['prefill'],dt,pt,L*self.sync_s_layer/dt,L*self.dispatch_s_layer/pt])
        x=np.asarray(result,dtype=np.float32)
        assert x.shape[1]==32 and np.isfinite(x).all() and (x>=0).all()
        return x
