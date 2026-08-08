# word2vec from scratch

A from-scratch implementation of skip-gram with negative sampling (Mikolov et al. 2013), trained on the `text8` corpus and evaluated on the Google word analogy task.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy
gcc -O3 -march=native -fPIC -shared -o word2vec_core.so word2vec_core.c -lm
python train.py
```

## Design

- **Core training loop in C** (`word2vec_core.c`), compiled to a shared library and called from Python via `ctypes`.
- Skip-gram with negative sampling (NEG), subsampling of frequent words, and linear learning rate decay.
- Vectors: input + output averaged as final embeddings.

## Files

- `word2vec_core.c` / `.so` — C training loop
- `train.py` — Python driver: vocabulary, subsampling table, negative table, evaluation, output
- `artifacts/vectors.txt` — final word vectors
- `artifacts/results.json` — accuracy metrics and hyperparameters

## Results

See `artifacts/results.json` for the latest measured numbers.