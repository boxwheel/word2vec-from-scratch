#!/usr/bin/env python3
"""
word2vec — Skip-gram with Negative Sampling, from scratch.

References:
  - Mikolov et al. (2013), "Efficient Estimation of Word Representations in
    Vector Space" (arXiv:1301.3781)
  - Mikolov et al. (2013), "Distributed Representations of Words and Phrases
    and their Compositionality" (arXiv:1310.4546)
"""

import sys
import os
import time
import json
import struct
import ctypes
import heapq
import argparse
from collections import Counter
from pathlib import Path

import numpy as np


# ── C library binding ────────────────────────────────────────────────────────

LIB = ctypes.CDLL(str(Path(__file__).resolve().parent / "word2vec_core.so"))

LIB.train_sg_neg.argtypes = [
    ctypes.c_void_p,          # train_data (int32*)
    ctypes.c_int64,           # train_words
    ctypes.c_void_p,          # W_in (float*)
    ctypes.c_void_p,          # W_out (float*)
    ctypes.c_int32,           # vocab_size
    ctypes.c_int32,           # dim
    ctypes.c_int32,           # window
    ctypes.c_void_p,          # subsample_table (float*, can be NULL)
    ctypes.c_float,           # subsample_threshold
    ctypes.c_void_p,          # neg_table (int32*)
    ctypes.c_int32,           # neg_table_size
    ctypes.c_int32,           # neg_samples
    ctypes.c_float,           # learning_rate
    ctypes.c_int32,           # epochs
    ctypes.c_uint32,          # seed
    ctypes.c_void_p,          # progress (int64*)
]
LIB.train_sg_neg.restype = ctypes.c_int


# ── Vocabulary building ──────────────────────────────────────────────────────

def build_vocab(filepath: str, min_count: int = 5):
    """Read text, count words, build vocabulary."""
    print(f"Building vocabulary from {filepath} (min_count={min_count})...")
    with open(filepath, "r") as f:
        text = f.read()

    words = text.split()
    del text

    counter = Counter(words)
    # Filter by min_count, keep in descending frequency order
    vocab_words = [(w, c) for w, c in counter.items() if c >= min_count]
    vocab_words.sort(key=lambda x: -x[1])

    word2id = {w: i for i, (w, _) in enumerate(vocab_words)}
    id2word = [w for w, _ in vocab_words]
    counts = [c for _, c in vocab_words]

    print(f"  Total tokens: {len(words):,}")
    print(f"  Unique tokens: {len(counter):,}")
    print(f"  Vocabulary size: {len(vocab_words):,} (min_count >= {min_count})")

    del words
    return word2id, id2word, counts


def text_to_ids(filepath: str, word2id: dict, unk_id: int | None = None):
    """Convert text file to a list of token IDs. Returns a bytes object for ctypes."""
    print(f"Converting text to token IDs from {filepath}...")
    with open(filepath, "r") as f:
        text = f.read()
    words = text.split()
    del text

    ids = []
    if unk_id is not None:
        for w in words:
            ids.append(word2id.get(w, unk_id))
    else:
        for w in words:
            wid = word2id.get(w)
            if wid is not None:
                ids.append(wid)

    del words
    print(f"  Token IDs: {len(ids):,} (dropped {len(ids) - sum(1 for x in ids if x >= 0)} OOVs)" 
          if unk_id is None else f"  Token IDs: {len(ids):,}")
    return np.array(ids, dtype=np.int32)


# ── Subsampling ──────────────────────────────────────────────────────────────

def build_subsample_table(counts, total_tokens, threshold=1e-5):
    """Build a subsampling probability table.
    
    P(discard) = 1 - sqrt(threshold / freq)
    We store P(keep) = 1 - P(discard) for fast check.
    """
    freq = np.array(counts, dtype=np.float64) / total_tokens
    # P(keep) = (sqrt(freq/threshold) + 1) * (threshold/freq)  ... 
    # Actually the paper formula: P(keep) = (sqrt(z/t) + 1) * (t/z)
    # where z = freq, t = threshold
    # Simplified: P(keep) = sqrt(t/z) + t/z ... no wait.
    # 
    # Paper: "discard the word with probability P(w_i) = 1 - sqrt(t / f(w_i))"
    # So P(keep) = sqrt(threshold / freq)
    # But we also clamp to 1.0 when freq is very low.
    keep_prob = np.sqrt(threshold / freq)
    keep_prob = np.minimum(keep_prob, 1.0)
    return keep_prob.astype(np.float32)


# ── Negative sampling table ──────────────────────────────────────────────────

def build_neg_table(counts, table_size=100_000_000):
    """Build unigram distribution raised to 3/4 power."""
    print(f"Building negative sampling table (size={table_size})...")
    pow_counts = np.array(counts, dtype=np.float64) ** 0.75
    total = pow_counts.sum()
    probs = pow_counts / total

    # Build table via sampling proportional to probs
    cumsum = np.cumsum(probs * table_size)
    cumsum = np.round(cumsum).astype(np.int64)

    table = np.zeros(table_size, dtype=np.int32)
    idx = 0
    for word_id, end in enumerate(cumsum):
        if end > idx:
            table[idx:end] = word_id
            idx = end
    # Fill any remaining slots with last word
    if idx < table_size:
        table[idx:] = len(counts) - 1

    return table


# ── Vector initialization ────────────────────────────────────────────────────

def init_vectors(vocab_size, dim, seed):
    """Initialize vectors with small random values in [-0.5/dim, 0.5/dim]."""
    rng = np.random.RandomState(seed)
    W_in = rng.uniform(-0.5 / dim, 0.5 / dim, (vocab_size, dim)).astype(np.float32)
    W_out = np.zeros((vocab_size, dim), dtype=np.float32)  # paper initializes output to zeros
    return W_in, W_out


# ── Evaluation ───────────────────────────────────────────────────────────────

def load_questions(filepath):
    """Load the Google analogy questions file.
    
    Returns list of (section, [quads]) where each quad is (a, b, c, d_expected).
    """
    sections = []
    current_section = None
    current_quads = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                if current_section and current_quads:
                    sections.append((current_section, current_quads))
                current_section = line.lstrip(": ")
                current_quads = []
            else:
                parts = line.split()
                if len(parts) == 4:
                    current_quads.append(tuple(p.lower() for p in parts))

    if current_section and current_quads:
        sections.append((current_section, current_quads))

    return sections


def evaluate_analogy(W, word2id, id2word, questions_file):
    """Evaluate word analogy accuracy.
    
    For a:b :: c:d, find word with vector closest to vec(b) - vec(a) + vec(c).
    Exclude a, b, c from candidates.
    """
    print(f"\nEvaluating analogies from {questions_file}...")
    sections = load_questions(questions_file)

    # Normalize all vectors for cosine similarity
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    W_norm = W / norms

    semantic_sections = [
        "capital-common-countries", "capital-world", "currency",
        "city-in-state", "family",
    ]
    # Broadly: first 5 sections are semantic, rest syntactic
    total_correct = 0
    total_asked = 0
    total_skipped = 0
    semantic_correct = 0
    semantic_asked = 0
    semantic_skipped = 0
    syntactic_correct = 0
    syntactic_asked = 0
    syntactic_skipped = 0

    for section_name, quads in sections:
        correct = 0
        asked = 0
        skipped = 0

        for a, b, c, expected in quads:
            if a not in word2id or b not in word2id or c not in word2id or expected not in word2id:
                skipped += 1
                continue

            id_a, id_b, id_c, id_d = word2id[a], word2id[b], word2id[c], word2id[expected]

            # vec(b) - vec(a) + vec(c)
            query = W_norm[id_b] - W_norm[id_a] + W_norm[id_c]

            # Cosine similarity with all words
            scores = np.dot(W_norm, query)

            # Exclude a, b, c
            scores[id_a] = -np.inf
            scores[id_b] = -np.inf
            scores[id_c] = -np.inf

            predicted = int(np.argmax(scores))
            asked += 1
            if predicted == id_d:
                correct += 1

        is_semantic = any(s in section_name.lower() for s in semantic_sections)
        total_correct += correct
        total_asked += asked
        total_skipped += skipped

        cat_acc = 100.0 * correct / asked if asked > 0 else 0.0

        if is_semantic:
            semantic_correct += correct
            semantic_asked += asked
            semantic_skipped += skipped
        else:
            syntactic_correct += correct
            syntactic_asked += asked
            syntactic_skipped += skipped

        print(f"  {section_name}: {correct}/{asked} = {cat_acc:.2f}%  (skipped {skipped})")

    total_acc = 100.0 * total_correct / total_asked if total_asked > 0 else 0.0
    sem_acc = 100.0 * semantic_correct / semantic_asked if semantic_asked > 0 else 0.0
    syn_acc = 100.0 * syntactic_correct / syntactic_asked if syntactic_asked > 0 else 0.0

    print(f"\n  TOTAL:     {total_correct}/{total_asked} = {total_acc:.2f}%  (skipped {total_skipped})")
    print(f"  Semantic:  {semantic_correct}/{semantic_asked} = {sem_acc:.2f}%  (skipped {semantic_skipped})")
    print(f"  Syntactic: {syntactic_correct}/{syntactic_asked} = {syn_acc:.2f}%  (skipped {syntactic_skipped})")

    return {
        "total": {"correct": total_correct, "asked": total_asked, "skipped": total_skipped,
                   "accuracy": round(total_acc, 4)},
        "semantic": {"correct": semantic_correct, "asked": semantic_asked, "skipped": semantic_skipped,
                      "accuracy": round(sem_acc, 4)},
        "syntactic": {"correct": syntactic_correct, "asked": syntactic_asked, "skipped": syntactic_skipped,
                       "accuracy": round(syn_acc, 4)},
    }


# ── Save vectors in word2vec text format ─────────────────────────────────────

def save_vectors(filepath, W_final, id2word):
    """Save vectors in word2vec text format.
    
    Line 1: <vocab_size> <dim>
    Then: word v1 v2 ... v_dim
    """
    vocab_size, dim = W_final.shape
    print(f"\nSaving vectors to {filepath}...")
    with open(filepath, "w") as f:
        f.write(f"{vocab_size} {dim}\n")
        for i, word in enumerate(id2word):
            vec_str = " ".join(f"{v:.6f}" for v in W_final[i])
            f.write(f"{word} {vec_str}\n")
    print(f"  Saved {vocab_size} vectors of dimension {dim}")


# ── Main training ────────────────────────────────────────────────────────────

def train(args):
    start_time = time.time()

    # 1. Build vocabulary
    word2id, id2word, counts = build_vocab(args.corpus, args.min_count)
    vocab_size = len(word2id)
    total_tokens = sum(counts)

    # 2. Convert text to IDs
    train_ids = text_to_ids(args.corpus, word2id, unk_id=None)
    train_size = len(train_ids)

    # 3. Build subsample table
    subsample_table = None
    if args.subsample > 0:
        subsample_table = build_subsample_table(counts, total_tokens, args.subsample)
        # Compute expected tokens kept
        keep_frac = subsample_table.mean()
        print(f"  Subsampling threshold: {args.subsample}, expected keep: {keep_frac:.2%}")
        print(f"  Expected training tokens: {int(train_size * keep_frac):,}")

    # 4. Build negative sampling table
    neg_table = build_neg_table(counts, args.neg_table_size)
    print(f"  Negative sampling table built: {len(neg_table):,} entries")

    # 5. Initialize vectors
    W_in, W_out = init_vectors(vocab_size, args.dim, args.seed)
    print(f"  Vectors initialized: {vocab_size} x {args.dim}")

    # 6. Training
    print(f"\n{'='*60}")
    print(f"Training skip-gram with negative sampling")
    print(f"  Vocab: {vocab_size}, Tokens: {train_size:,}, Dim: {args.dim}")
    print(f"  Window: {args.window}, Neg: {args.neg}, LR: {args.lr}")
    print(f"  Epochs: {args.epochs}, Seed: {args.seed}")
    print(f"{'='*60}")

    progress = ctypes.c_int64(0)
    total_steps = train_size * args.epochs

    # Pin arrays to get contiguous C pointers
    train_ids_c = np.ascontiguousarray(train_ids, dtype=np.int32)
    W_in_c = np.ascontiguousarray(W_in, dtype=np.float32)
    W_out_c = np.ascontiguousarray(W_out, dtype=np.float32)
    subsample_c = np.ascontiguousarray(subsample_table, dtype=np.float32) if subsample_table is not None else None
    neg_table_c = np.ascontiguousarray(neg_table, dtype=np.int32)

    # Report progress periodically
    import threading
    done = [False]

    def progress_reporter():
        last = -1
        while not done[0]:
            cur = progress.value
            if cur != last and cur > 0:
                pct = 100.0 * cur / total_steps
                elapsed = time.time() - start_time
                if cur > 0 and elapsed > 0:
                    wps = cur / elapsed
                    eta = (total_steps - cur) / wps if wps > 0 else 0
                    print(f"\r  Progress: {pct:.1f}% ({cur:,}/{total_steps:,}) "
                          f"{wps:,.0f} words/sec, ETA: {eta:.0f}s  ", end="", flush=True)
                last = cur
            time.sleep(2)

    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    try:
        ret = LIB.train_sg_neg(
            train_ids_c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(train_size),
            W_in_c.ctypes.data_as(ctypes.c_void_p),
            W_out_c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(vocab_size),
            ctypes.c_int32(args.dim),
            ctypes.c_int32(args.window),
            subsample_c.ctypes.data_as(ctypes.c_void_p) if subsample_c is not None else None,
            ctypes.c_float(args.subsample),
            neg_table_c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(len(neg_table)),
            ctypes.c_int32(args.neg),
            ctypes.c_float(args.lr),
            ctypes.c_int32(args.epochs),
            ctypes.c_uint32(args.seed),
            ctypes.byref(progress),
        )
    finally:
        done[0] = True
        reporter.join(timeout=3)

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.1f}s ({training_time/60:.1f}m)")

    # 7. Combine W_in and W_out (use W_in + W_out as final vectors, common practice)
    W_final = W_in_c + W_out_c

    # 8. Evaluate
    eval_results = evaluate_analogy(W_final, word2id, id2word, args.questions)

    # 9. Save vectors
    save_vectors(args.output, W_final, id2word)

    # 10. Save results
    results = {
        "hyperparameters": {
            "dim": args.dim,
            "window": args.window,
            "negative_samples": args.neg,
            "epochs": args.epochs,
            "subsample_threshold": args.subsample,
            "learning_rate": args.lr,
            "min_count": args.min_count,
            "seed": args.seed,
            "neg_table_size": args.neg_table_size,
            "training_time_seconds": round(training_time, 1),
        },
        "data": {
            "corpus": args.corpus,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "training_tokens": train_size,
        },
        "accuracy": eval_results,
    }

    results_path = args.results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train word2vec skip-gram with negative sampling")
    parser.add_argument("--corpus", default="data/text8", help="Path to text corpus")
    parser.add_argument("--questions", default="data/questions-words.txt", help="Path to analogy questions")
    parser.add_argument("--output", default="artifacts/vectors.txt", help="Output vector file")
    parser.add_argument("--results", default="artifacts/results.json", help="Output results JSON")
    parser.add_argument("--dim", type=int, default=100, help="Vector dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--neg", type=int, default=5, help="Negative samples")
    parser.add_argument("--lr", type=float, default=0.025, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--subsample", type=float, default=1e-5,
                        help="Subsampling threshold (0 to disable)")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum word frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--neg-table-size", type=int, default=100_000_000,
                        help="Negative sampling table size")
    args = parser.parse_args()
    train(args)