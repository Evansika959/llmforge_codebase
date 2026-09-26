/* Fixed-workload timing wrapper. The production kernel source is unchanged.
 * All timestamps share CLOCK_MONOTONIC. No temperature polling in inference.
 * Model loading is outside the baseline and timed inference windows.
 */
#define main original_runq_main
#include "../../../src/runq_llmforge.c"
#undef main
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>

static atomic_int lw_done = 0;
static const char *lw_node;
static FILE *lw_trace;
static double lw_deadline;
static double lw_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int lw_read(const char *name, long long *value) {
    char path[512]; snprintf(path, sizeof(path), "%s/%s", lw_node, name);
    FILE *f = fopen(path, "r"); if (!f) return -1;
    int ok = fscanf(f, "%lld", value); fclose(f); return ok == 1 ? 0 : -1;
}
static void lw_guard(void) {
    long long b;
    if (lw_read("capacity", &b) || b <= 30 || b > 100) {
        fprintf(stderr, "BATTERY_STOP: charge above 30%% and unplug before resume\n");
        fflush(stderr); _exit(75);
    }
    char path[512], status[80] = {0};
    snprintf(path, sizeof(path), "%s/status", lw_node);
    FILE *f = fopen(path, "r");
    if (!f || !fgets(status, sizeof(status), f)) {
        if (f) fclose(f);
        fprintf(stderr, "CHARGING_TELEMETRY_UNAVAILABLE\n"); fflush(stderr); _exit(75);
    }
    fclose(f);
    if (strncmp(status, "Discharging", 11) && strncmp(status, "Not charging", 12)) {
        fprintf(stderr, "CHARGER_STOP\n"); fflush(stderr); _exit(75);
    }
}
static void *lw_sample(void *unused) {
    (void)unused; int index = 0;
    while (!atomic_load(&lw_done)) {
        if (!(index++ % 10)) lw_guard();
        if (lw_now() > lw_deadline) {
            fprintf(stderr, "MEASUREMENT_TIMEOUT\n"); fflush(stderr); _exit(78);
        }
        long long current, voltage;
        if (!lw_read("current_now", &current) && !lw_read("voltage_now", &voltage)) {
            fprintf(lw_trace, "%.9f,%lld,%lld\n", lw_now(), current, voltage);
            fflush(lw_trace);
        }
        struct timespec delay = {0, 100000000}; nanosleep(&delay, NULL);
    }
    return NULL;
}
static void lw_stop(int sig) { _exit(128 + sig); }

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: lw_measure MODEL TOKENIZER PROMPT POWER_NODE TEMP_NODE TRACE\n");
        return 2;
    }
    signal(SIGTERM, lw_stop); signal(SIGINT, lw_stop);
    lw_node = argv[4]; lw_guard(); lw_deadline = lw_now() + 300.;
    lw_trace = fopen(argv[6], "wx"); if (!lw_trace) return 3;
    pthread_t sampler_thread;
    if (pthread_create(&sampler_thread, NULL, lw_sample, NULL)) return 4;
    Transformer t; build_transformer(&t, argv[1]);
    GPT2Tokenizer tok = gpt2_load(argv[2]);
    int ids[512], count = 0;
    if (strlen(argv[3]) >= 500) return 5;
    gpt2_encode(&tok, argv[3], ids, &count);
    if (count != 49 || t.config.seq_len < 80) {
        fprintf(stderr, "WORKLOAD_MISMATCH prompt=%d\n", count); return 6;
    }
    Sampler sampler; build_sampler(&sampler, t.config.vocab_size, 0.8f, 0.9f, 42);
    double idle_start = lw_now(); sleep(4);
    FILE *tf = fopen(argv[5], "r"); long temp = 0;
    if (!tf || fscanf(tf, "%ld", &temp) != 1 || temp <= 0) {
        if (tf) fclose(tf); fprintf(stderr, "TEMPERATURE_UNAVAILABLE\n"); return 77;
    }
    fclose(tf);
    if (temp >= 45000) { fprintf(stderr, "THERMAL_ADMISSION %.1fC\n", temp / 1000.); return 76; }
    lw_guard();
    /* Exactly 32 generated tokens, even for a synthetic EOS; 31 decode forwards. */
    double start = lw_now();
    float *logits = prefill_prompt(&t, ids, count);
    int token = sample(&sampler, logits);
    double prefill_end = lw_now();
    for (int step = 0; step < 31; step++) {
        logits = forward(&t, token, count + step, 1);
        token = sample(&sampler, logits);
    }
    double end = lw_now();
    sleep(4); double post_end = lw_now();
    atomic_store(&lw_done, 1); pthread_join(sampler_thread, NULL); fclose(lw_trace);
    printf("{\"idle_start\":%.9f,\"start\":%.9f,\"prefill_end\":%.9f,\"end\":%.9f,\"post_end\":%.9f,"
           "\"prefill_tokens\":49,\"output_tokens\":32,\"decode_tokens\":31,\"admission_temperature_c\":%.3f,\"last_token\":%d}\n",
           idle_start,start,prefill_end,end,post_end,temp/1000.,token);
    free_sampler(&sampler); gpt2_free(&tok); free_transformer(&t);
    return 0;
}
