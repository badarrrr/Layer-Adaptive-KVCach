# WikiText-2 Test Results

- Model: `llama2_7b`
- Test samples: `64`
- Sampling seed: `6520`
- Maximum tokens per sample: `256`

| Method | Requested | Actual | PPL | Relative PPL | Avg cache (MB) | Peak cache (MB) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| No compression | 0% | 0.0% | 10.083 | 0.00% | 45.88 | 127.50 | 54.75 |
| Uniform recent | 10% | 9.6% | 882.657 | 8653.61% | 41.48 | 115.00 | 53.94 |
| Uniform heavy-hitter | 10% | 9.6% | 887.157 | 8698.23% | 41.48 | 115.00 | 53.08 |
| Layer-adaptive | 10% | 14.4% | 292.555 | 2801.37% | 39.29 | 108.50 | 52.86 |
| Uniform recent | 20% | 19.1% | 1073.368 | 10544.95% | 37.14 | 102.00 | 53.63 |
| Uniform heavy-hitter | 20% | 19.1% | 876.220 | 8589.77% | 37.14 | 102.00 | 52.06 |
| Layer-adaptive | 20% | 29.4% | 325.928 | 3132.34% | 32.41 | 88.00 | 51.12 |
| Uniform recent | 30% | 28.4% | 1244.993 | 12247.02% | 32.85 | 89.00 | 53.25 |
| Uniform heavy-hitter | 30% | 28.4% | 909.064 | 8915.49% | 32.85 | 89.00 | 51.09 |
| Layer-adaptive | 30% | 45.7% | 398.589 | 3852.94% | 24.93 | 64.50 | 49.67 |

Note: Perplexity is measured on a fixed, seeded WikiText-2 test subset with context reset for each passage. The current Transformers-compatible implementation keeps a uniform KV-cache length across layers while using layer-specific token-selection policies.
