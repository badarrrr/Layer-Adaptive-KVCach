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

The profiling script measures:

- baseline incremental perplexity
- per-layer sensitivity curves under KV cache pruning
- attention top-token overlap as a redundancy score
- safe compression threshold per layer

Example:

```powershell
$env:PYTHONPATH="src"
python scripts/profile_layers.py `
  --model mistral_7b `
  --output outputs/profiles/mistral_7b_profile.json `
  --samples 32 `
  --max-length 512
```

For Llama 2 7B, make sure your Hugging Face account has access to the gated model:

```powershell
$env:PYTHONPATH="src"
python scripts/profile_layers.py `
  --model llama2_7b `
  --output outputs/profiles/llama2_7b_profile.json
```

## Runtime Generation

Example:

```powershell
$env:PYTHONPATH="src"
python scripts/generate.py `
  --model mistral_7b `
  --profile outputs/profiles/mistral_7b_profile.json `
  --prompt "Explain why KV cache memory grows during autoregressive decoding." `
  --global-compression 0.3 `
  --max-new-tokens 128
```

## Method Summary

The runtime scheduler allocates a user-provided global compression budget across layers. Shallow layers use conservative heavy-hitter pruning, middle layers use a hybrid recent-token and attention-based policy, and deep layers use a more aggressive SnapKV-style policy. The code does not modify model weights or architecture; it only changes the retained sequence positions in `past_key_values` during inference.
