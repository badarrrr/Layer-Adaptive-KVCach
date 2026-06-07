# WikiText-2 Test Results

- Model: `llama2_7b`
- Test samples: `64`
- Sampling seed: `6520`
- Maximum tokens per sample: `256`

| Method | Requested | Actual | PPL | Relative PPL | Avg cache (MB) | Peak cache (MB) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| No compression | 0% | 0.0% | 10.083 | 0.00% | 45.88 | 127.50 | 54.82 |
| Uniform recent | 2% | 1.9% | 10.120 | 0.37% | 45.00 | 125.00 | 52.86 |
| Uniform heavy-hitter | 2% | 1.9% | 10.086 | 0.02% | 45.00 | 125.00 | 53.29 |
| Layer-adaptive | 2% | 1.9% | 10.083 | -0.01% | 45.00 | 125.00 | 54.02 |
| Uniform recent | 5% | 4.8% | 10.197 | 1.13% | 43.68 | 121.00 | 53.76 |
| Uniform heavy-hitter | 5% | 4.8% | 10.094 | 0.10% | 43.68 | 121.00 | 52.00 |
| Layer-adaptive | 5% | 4.8% | 10.091 | 0.08% | 43.68 | 121.00 | 53.13 |
| Uniform recent | 10% | 9.6% | 10.375 | 2.89% | 41.48 | 115.00 | 53.20 |
| Uniform heavy-hitter | 10% | 9.6% | 10.119 | 0.36% | 41.48 | 115.00 | 52.98 |
| Layer-adaptive | 10% | 9.6% | 10.119 | 0.36% | 41.48 | 115.00 | 53.42 |
| Uniform recent | 20% | 19.1% | 10.622 | 5.34% | 37.14 | 102.00 | 53.03 |
| Uniform heavy-hitter | 20% | 19.1% | 10.202 | 1.18% | 37.14 | 102.00 | 51.61 |
| Layer-adaptive | 20% | 19.1% | 10.195 | 1.11% | 37.14 | 102.00 | 51.38 |

Note: Perplexity is measured on a fixed, seeded WikiText-2 test subset with context reset for each passage. The current Transformers-compatible implementation keeps a uniform physical KV-cache length derived from the requested global compression ratio while using layer-specific token-selection policies. Throughput includes attention collection and Python compression overhead.
