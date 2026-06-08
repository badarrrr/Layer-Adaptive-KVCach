# WikiText-2 Test Results

- Model: `mistral_7b`
- Test samples: `64`
- Sampling seed: `6520`
- Maximum tokens per sample: `256`

| Method | Requested | Actual | PPL | Relative PPL | Avg cache (MB) | Peak cache (MB) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| No compression | 0% | 0.0% | 9.887 | 0.00% | 11.37 | 31.88 | 59.25 |
| Uniform recent | 2% | 1.9% | 10.119 | 2.34% | 11.15 | 31.25 | 58.62 |
| Uniform heavy-hitter | 2% | 1.9% | 10.075 | 1.89% | 11.15 | 31.25 | 59.09 |
| Layer-adaptive | 2% | 1.9% | 10.076 | 1.91% | 11.15 | 31.25 | 59.10 |
| Uniform recent | 5% | 4.8% | 10.430 | 5.48% | 10.82 | 30.25 | 58.32 |
| Uniform heavy-hitter | 5% | 4.8% | 10.357 | 4.75% | 10.82 | 30.25 | 58.18 |
| Layer-adaptive | 5% | 4.8% | 10.358 | 4.76% | 10.82 | 30.25 | 57.94 |
| Uniform recent | 10% | 9.6% | 11.466 | 15.96% | 10.28 | 28.75 | 58.45 |
| Uniform heavy-hitter | 10% | 9.6% | 11.177 | 13.04% | 10.28 | 28.75 | 57.89 |
| Layer-adaptive | 10% | 9.6% | 11.176 | 13.03% | 10.28 | 28.75 | 57.81 |
| Uniform recent | 20% | 19.0% | 13.568 | 37.22% | 9.20 | 25.50 | 57.55 |
| Uniform heavy-hitter | 20% | 19.0% | 12.952 | 30.99% | 9.20 | 25.50 | 56.29 |
| Layer-adaptive | 20% | 19.0% | 12.959 | 31.06% | 9.20 | 25.50 | 55.83 |

Note: Perplexity is measured on a fixed, seeded WikiText-2 test subset with context reset for each passage. The current Transformers-compatible implementation keeps a uniform physical KV-cache length derived from the requested global compression ratio while using layer-specific token-selection policies. Throughput includes attention collection and Python compression overhead.
