/* Build against both the archived baseline and the optimized source. */
#define main kernel_original_main
#ifndef NLF_TEST_KERNEL
#define NLF_TEST_KERNEL "../../../../src/runq_llmforge.c"
#endif
#include NLF_TEST_KERNEL
#undef main

static uint32_t rng = 1234567;
static int8_t next_q(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (int8_t)rng; }
static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void reference(float *out, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int row = 0; row < d; row++) {
        float value = 0;
        for (int j = 0; j < n; j += GS) {
            int32_t dot = 0;
            for (int k = 0; k < GS; k++) dot += (int32_t)x->q[j+k] * (int32_t)w->q[row*n+j+k];
            value += (float)dot * w->s[(row*n+j)/GS] * x->s[j/GS];
        }
        out[row] = value;
    }
}
static int check(int n, int d, int pattern, int bench) {
    QuantizedTensor x = {malloc(n), malloc((n/GS)*sizeof(float))};
    QuantizedTensor w = {malloc((size_t)n*d), malloc((size_t)n*d/GS*sizeof(float))};
    float *out = malloc(d*sizeof(float)), *ref = malloc(d*sizeof(float));
    if (!x.q || !x.s || !w.q || !w.s || !out || !ref) return 2;
    for (int i=0; i<n; i++) x.q[i] = pattern==0 ? next_q() : pattern==1 ? -128 : pattern==2 ? 127 : 0;
    for (int i=0; i<n*d; i++) w.q[i] = pattern==0 ? next_q() : pattern==1 ? -128 : pattern==2 ? (i%2 ? -128 : 127) : next_q();
    for (int i=0; i<n/GS; i++) x.s[i] = (i%5+1)*0.000125f;
    for (int i=0; i<n*d/GS; i++) w.s[i] = (i%7+1)*0.00025f;
    reference(ref,&x,&w,n,d); matmul(out,&x,&w,n,d);
    for (int i=0; i<d; i++) {
        if (!isfinite(out[i]) || fabsf(out[i]-ref[i]) > 1e-6f + 2e-5f*fabsf(ref[i])) {
            fprintf(stderr,"FAIL gs=%d n=%d d=%d pattern=%d row=%d actual=%.9g ref=%.9g\n",GS,n,d,pattern,i,out[i],ref[i]);
            return 1;
        }
    }
    if (bench) {
        for (int r=0; r<3; r++) {
            double begin=seconds();
            for (int it=0; it<20; it++) matmul(out,&x,&w,n,d);
            printf("{\"gs\":%d,\"n\":%d,\"d\":%d,\"repeat\":%d,\"ms\":%.6f}\n",GS,n,d,r,1000*(seconds()-begin)/20);
        }
    }
    free(x.q);free(x.s);free(w.q);free(w.s);free(out);free(ref);return 0;
}
int main(int argc, char **argv) {
    if (argc==4 && !strcmp(argv[1],"--logits")) {
        Transformer t; build_transformer(&t,argv[2]);
        int ids[49]; for(int i=0;i<49;i++) ids[i]=i % t.config.vocab_size;
        FILE *f=fopen(argv[3],"wb"); if(!f) return 2;
        float *logits=prefill_prompt(&t,ids,49);
        Sampler sampler; build_sampler(&sampler,t.config.vocab_size,.8f,.9f,42);
        for(int step=0;step<32;step++) {
            for(int i=0;i<t.config.vocab_size;i++) if(!isfinite(logits[i])) return 3;
            if(fwrite(logits,sizeof(float),t.config.vocab_size,f)!=(size_t)t.config.vocab_size) return 4;
            int token=sample(&sampler,logits);
            printf("%d\n",token);
            if(step<31) logits=forward(&t,token,49+step,1);
        }
        fclose(f);free_sampler(&sampler);free_transformer(&t);return 0;
    }
    int bench=argc==2 && !strcmp(argv[1],"--bench");
    int widths[]={64,192,256,384,576,640,768,960,1152,1280,1536,1920,2560};
    for(GS=16;GS<=64;GS*=2) {
        if(bench) {
            if(GS!=64) continue;
            int shapes[][2]={{576,50257},{960,50257},{576,1536},{960,2560},{384,576}};
            for(int s=0;s<5;s++) if(check(shapes[s][0],shapes[s][1],0,1)) return 1;
        } else {
            for(int s=0;s<13;s++) for(int pattern=0;pattern<4;pattern++)
                if(check(widths[s],37,pattern,0)) return 1;
        }
    }
    fprintf(stderr,"PASS GS16/32/64 matrix products (random, -128, alternating extrema, zero)\n");
    return 0;
}
