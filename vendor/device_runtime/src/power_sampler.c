#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static volatile sig_atomic_t keep_running = 1;


static void stop_sampling(int signal_number) {
    (void)signal_number;
    keep_running = 0;
}


static int read_integer(const char* path, long long* value) {
    FILE* file = fopen(path, "r");
    if (file == NULL) return -1;
    int matched = fscanf(file, "%lld", value);
    fclose(file);
    return matched == 1 ? 0 : -1;
}


static double elapsed_seconds(struct timespec start, struct timespec now) {
    return (double)(now.tv_sec - start.tv_sec)
        + (double)(now.tv_nsec - start.tv_nsec) / 1000000000.0;
}


static void sleep_milliseconds(long interval_ms) {
    struct timespec request = {
        .tv_sec = interval_ms / 1000,
        .tv_nsec = (interval_ms % 1000) * 1000000L,
    };
    while (keep_running && nanosleep(&request, &request) != 0 && errno == EINTR) {
    }
}


int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s POWER_SUPPLY_DIR DURATION_SEC INTERVAL_MS OUTPUT_CSV\n", argv[0]);
        return EXIT_FAILURE;
    }

    char current_path[PATH_MAX];
    char voltage_path[PATH_MAX];
    if (snprintf(current_path, sizeof(current_path), "%s/current_now", argv[1]) >= (int)sizeof(current_path)
        || snprintf(voltage_path, sizeof(voltage_path), "%s/voltage_now", argv[1]) >= (int)sizeof(voltage_path)) {
        fprintf(stderr, "power-supply path is too long\n");
        return EXIT_FAILURE;
    }

    char* parse_end = NULL;
    double duration_sec = strtod(argv[2], &parse_end);
    if (parse_end == argv[2] || *parse_end != '\0' || duration_sec <= 0.0) {
        fprintf(stderr, "invalid duration: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    parse_end = NULL;
    long interval_ms = strtol(argv[3], &parse_end, 10);
    if (parse_end == argv[3] || *parse_end != '\0' || interval_ms <= 0) {
        fprintf(stderr, "invalid interval: %s\n", argv[3]);
        return EXIT_FAILURE;
    }

    long long probe = 0;
    if (read_integer(current_path, &probe) != 0 || read_integer(voltage_path, &probe) != 0) {
        fprintf(stderr, "cannot read current_now and voltage_now under %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    FILE* output = fopen(argv[4], "w");
    if (output == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", argv[4], strerror(errno));
        return EXIT_FAILURE;
    }

    signal(SIGINT, stop_sampling);
    signal(SIGTERM, stop_sampling);

    struct timespec start;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fprintf(stderr, "clock_gettime failed: %s\n", strerror(errno));
        fclose(output);
        return EXIT_FAILURE;
    }

    while (keep_running) {
        struct timespec now;
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) break;
        double elapsed = elapsed_seconds(start, now);
        if (elapsed > duration_sec) break;

        long long current_ua = 0;
        long long voltage_uv = 0;
        if (read_integer(current_path, &current_ua) == 0
            && read_integer(voltage_path, &voltage_uv) == 0) {
            fprintf(output, "%.6f,%lld,%lld\n", elapsed, current_ua, voltage_uv);
            fflush(output);
        }
        sleep_milliseconds(interval_ms);
    }

    fclose(output);
    return EXIT_SUCCESS;
}
