# Layer-Adaptive KV Cache Compression

Prototype code for the project **Layer-Adaptive KV Cache Compression via Offline Prioritization and Runtime Scheduling**.

The implementation targets two experiment models:

- `llama2_7b`: `meta-llama/Llama-2-7b-hf`
- `mistral_7b`: `mistralai/Mistral-7B-v0.1`

## Structure

- `src/lakvc/`: reusable implementation
- `scripts/profile_layers.py`: offline layer-wise importance and redundancy profiling
- `scripts/generate.py`: online generation with runtime layer-adaptive KV compression
- `configs/models.json`: model IDs and loading settings

## Offline Profiling

On the Linux server, set `PYTHONPATH` before running scripts. The model and
dataset are expected to be available locally according to the existing project
configuration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

The profiling script measures:

- baseline incremental perplexity
- per-layer sensitivity curves under KV cache pruning
- attention top-token overlap as a redundancy score
- safe compression threshold per layer

Example:

```bash
export PYTHONPATH=src
python scripts/profile_layers.py \
  --model mistral_7b \
  --data-file datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --split train \
  --output outputs/profiles/mistral_7b_profile.json \
  --samples 32 \
  --max-length 512
```

For Llama 2 7B, make sure your Hugging Face account has access to the gated model:

```bash
export PYTHONPATH=src
python scripts/profile_layers.py \
  --model llama2_7b \
  --data-file datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --split train \
  --output outputs/profiles/llama2_7b_profile.json
```

Use `--data-file` for a local parquet shard, `--local-text-file` for one sample
per line, or `--dataset`/`--dataset-config`/`--split` for a Hugging Face dataset.
Use the training split as the offline calibration set, and reserve validation
and test splits for reporting final quality metrics.

## Runtime Generation

Example:

```bash
export PYTHONPATH=src
python scripts/generate.py \
  --model mistral_7b \
  --profile outputs/profiles/mistral_7b_profile.json \
  --prompt "Explain why KV cache memory grows during autoregressive decoding." \
  --global-compression 0.2 \
  --max-new-tokens 128 \
  --recent-window 64 \
  --sink-tokens 4 \
  --min-tokens 64 \
  --compression-start-tokens 64 \
  --max-layer-compression 0.35 \
  --repetition-penalty 1.05 \
  --no-repeat-ngram-size 4
```

The generation script formats prompts as `Question: ... Answer:` by default,
which is more reliable for base models such as `Llama-2-7b-hf`. Pass
`--raw-prompt` to disable this behavior.

## WikiText Test Evaluation

Evaluate the saved training-split profile on the held-out WikiText-2 test
split, comparing no compression, uniform recent-token pruning, uniform
heavy-hitter pruning, and layer-adaptive token selection:

```bash
export PYTHONPATH=src
python scripts/evaluate_wikitext.py \
  --model llama2_7b \
  --profile outputs/profiles/llama2_7b_profile_final.json \
  --data-file datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet \
  --split test \
  --samples 64 \
  --max-length 256 \
  --ratios 0.02 0.05 0.1 0.2 \
  --sink-tokens 4 \
  --output-dir outputs/wikitext_test
```

The output directory contains JSON and CSV measurements, a Markdown result
table, the exact evaluated samples, and PNG figures. Treat results as
preliminary until the small-ratio sanity checks and matched-compression
comparisons pass.

## Method Summary

Transformers 4.41 uses one causal mask across decoder layers on the supported
legacy-cache path, so all layers currently retain the same physical number of
tokens. That common length is derived directly from the requested global
compression ratio, ensuring fair comparison with uniform baselines. The
layer-adaptive scheduler changes which tokens each layer retains: shallow
layers use conservative attention-sink/heavy-hitter selection, middle layers
use a hybrid recent-token and attention-based policy, and deep layers use a
more aggressive SnapKV-inspired policy. The code does not modify model weights.
True heterogeneous per-layer cache lengths require model-level mask/cache
changes and are not claimed by this prototype.
