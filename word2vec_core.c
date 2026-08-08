/*
 * word2vec_core.c — Skip-gram training loop with negative sampling.
 * Compiled as a shared library, called from Python via ctypes.
 *
 * Compile:
 *   gcc -O3 -march=native -fPIC -shared -o word2vec_core.so word2vec_core.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* A simple xorshift PRNG (fast, inline, good enough for training). */
typedef struct {
    uint32_t state;
} rng_t;

static inline uint32_t rng_next(rng_t *r) {
    uint32_t x = r->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->state = x;
    return x;
}

static inline float rng_uniform(rng_t *r) {
    return (float)(rng_next(r) & 0xFFFFFF) / (float)0x1000000;
}

/* sigmoid, clamped for numerical stability. */
static inline float sigmoid(float x) {
    if (x > 6.0f) return 1.0f;
    if (x < -6.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/*
 * Train skip-gram with negative sampling.
 *
 * Parameters (all arrays are float* or int32_t*):
 *   train_data       — array of token indices (int32)
 *   train_words      — number of tokens in train_data
 *   W_in             — input (center) vectors, shape [vocab_size * dim], float
 *   W_out            — output (context) vectors, shape [vocab_size * dim], float
 *   vocab_size       — number of vocabulary items
 *   dim              — vector dimension
 *   window           — maximum context window size
 *   subsample_table  — precomputed subsampling table (same size as vocab)
 *   subsample_threshold — threshold for subsampling (unused if table is precomputed)
 *   neg_table        — unigram-based negative sampling table (int32)
 *   neg_table_size   — size of neg_table
 *   neg_samples      — number of negative samples per positive pair
 *   learning_rate    — initial learning rate
 *   epochs           — number of passes through the data
 *   seed             — PRNG seed
 *   progress         — output: set to current progress count (updated per 100k words)
 *
 * Returns 0 on success.
 */
int train_sg_neg(
    const int32_t *train_data,
    int64_t train_words,
    float *W_in,
    float *W_out,
    int32_t vocab_size,
    int32_t dim,
    int32_t window,
    const float *subsample_table,
    float subsample_threshold,
    const int32_t *neg_table,
    int32_t neg_table_size,
    int32_t neg_samples,
    float learning_rate,
    int32_t epochs,
    uint32_t seed,
    volatile int64_t *progress
) {
    rng_t rng;
    float lr;
    int64_t total_words = train_words * epochs;

    for (int32_t ep = 0; ep < epochs; ep++) {
        rng.state = seed + (uint32_t)(ep * 2654435761U);

        for (int64_t pos = 0; pos < train_words; pos++) {
            /* --- progress update --- */
            if ((pos & 0xFFFF) == 0) {  /* every 65536 tokens */
                *progress = (int64_t)ep * train_words + pos;
            }

            int32_t center = train_data[pos];
            if (center < 0 || center >= vocab_size) continue;

            /* subsampling check */
            if (subsample_table) {
                if (rng_uniform(&rng) > subsample_table[center]) continue;
            } else if (subsample_threshold > 0) {
                /* this path not used if table is precomputed */
            }

            /* dynamic window size: uniform random in [1, window] */
            int32_t win = (int32_t)(rng_uniform(&rng) * (float)window) + 1;

            /* learning rate decay */
            {
                int64_t global_pos = (int64_t)ep * train_words + pos;
                float progress_frac = (float)global_pos / (float)total_words;
                lr = learning_rate * (1.0f - progress_frac);
                if (lr < learning_rate * 0.0001f) lr = learning_rate * 0.0001f;
            }

            /* iterate over context window */
            int64_t ctx_start = pos - win;
            if (ctx_start < 0) ctx_start = 0;
            int64_t ctx_end = pos + win + 1;
            if (ctx_end > train_words) ctx_end = train_words;

            for (int64_t ctx = ctx_start; ctx < ctx_end; ctx++) {
                if (ctx == pos) continue;

                int32_t context = train_data[ctx];
                if (context < 0 || context >= vocab_size) continue;

                /* --- positive sample --- */
                {
                    float *v_center = W_in + (int64_t)center * dim;
                    float *u_context = W_out + (int64_t)context * dim;

                    /* dot product */
                    float dot = 0.0f;
                    for (int32_t d = 0; d < dim; d++) {
                        dot += v_center[d] * u_context[d];
                    }

                    float g = sigmoid(dot) - 1.0f;  /* gradient for positive */

                    /* update both vectors */
                    for (int32_t d = 0; d < dim; d++) {
                        float grad_v = lr * g * u_context[d];
                        float grad_u = lr * g * v_center[d];
                        v_center[d] -= grad_v;
                        u_context[d] -= grad_u;
                    }
                }

                /* --- negative samples --- */
                for (int32_t n = 0; n < neg_samples; n++) {
                    int32_t neg = neg_table[(int32_t)(rng_uniform(&rng) * (float)neg_table_size)];
                    if (neg < 0 || neg >= vocab_size) continue;
                    if (neg == center || neg == context) continue;

                    float *v_center = W_in + (int64_t)center * dim;
                    float *u_neg = W_out + (int64_t)neg * dim;

                    float dot = 0.0f;
                    for (int32_t d = 0; d < dim; d++) {
                        dot += v_center[d] * u_neg[d];
                    }

                    float g = sigmoid(dot);  /* gradient for negative (target=0) */

                    for (int32_t d = 0; d < dim; d++) {
                        float grad_v = lr * g * u_neg[d];
                        float grad_u = lr * g * v_center[d];
                        v_center[d] -= grad_v;
                        u_neg[d] -= grad_u;
                    }
                }
            }
        }
    }

    *progress = total_words;
    return 0;
}