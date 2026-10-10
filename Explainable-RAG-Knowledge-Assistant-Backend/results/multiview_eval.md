# Milestone 3 - Multi-view RRF evaluation (EXPLORATORY, not a benchmark)

Known-positive Hit@k/MRR@10 on unverified seed pairs (incomplete qrels; queries may have been written from the comments). Audit is an assistant draft pending human confirmation. No significance testing; small n.

RRF k=60, depth=50 per view. Audit labels: {'correct': 95, 'partial': 17, 'incorrect': 13, 'uncertain': 3}

## Positive set: seed_all

| Method | Sinhala-target n | Hit@1 | Hit@10 | MRR@10 | Latin-target n | Hit@1 | Hit@10 | MRR@10 | All n | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 original | 57 | 0.018 | 0.018 | 0.018 | 36 | 0.222 | 0.500 | 0.332 | 82 | 0.232 | 0.158 |
| BM25 normalized | 57 | 0.158 | 0.439 | 0.244 | 36 | 0.083 | 0.194 | 0.119 | 82 | 0.390 | 0.222 |
| RRF: BM25 orig+norm | 57 | 0.035 | 0.316 | 0.123 | 36 | 0.194 | 0.528 | 0.303 | 82 | 0.439 | 0.216 |
| LaBSE original | 57 | 0.000 | 0.053 | 0.009 | 36 | 0.056 | 0.167 | 0.086 | 82 | 0.110 | 0.044 |
| LaBSE normalized | 57 | 0.158 | 0.421 | 0.241 | 36 | 0.056 | 0.194 | 0.091 | 82 | 0.378 | 0.207 |
| RRF: LaBSE orig+norm | 57 | 0.035 | 0.298 | 0.116 | 36 | 0.056 | 0.250 | 0.108 | 82 | 0.317 | 0.128 |
| RRF: hybrid BM25+LaBSE orig+norm | 57 | 0.158 | 0.421 | 0.237 | 36 | 0.167 | 0.417 | 0.238 | 82 | 0.463 | 0.268 |

## Positive set: audit_correct

| Method | Sinhala-target n | Hit@1 | Hit@10 | MRR@10 | Latin-target n | Hit@1 | Hit@10 | MRR@10 | All n | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 original | 47 | 0.021 | 0.021 | 0.021 | 31 | 0.258 | 0.581 | 0.385 | 72 | 0.264 | 0.180 |
| BM25 normalized | 47 | 0.192 | 0.532 | 0.296 | 31 | 0.097 | 0.226 | 0.138 | 72 | 0.444 | 0.253 |
| RRF: BM25 orig+norm | 47 | 0.043 | 0.383 | 0.149 | 31 | 0.226 | 0.613 | 0.352 | 72 | 0.500 | 0.246 |
| LaBSE original | 47 | 0.000 | 0.064 | 0.011 | 31 | 0.065 | 0.194 | 0.100 | 72 | 0.125 | 0.050 |
| LaBSE normalized | 47 | 0.192 | 0.468 | 0.283 | 31 | 0.065 | 0.226 | 0.106 | 72 | 0.403 | 0.230 |
| RRF: LaBSE orig+norm | 47 | 0.043 | 0.362 | 0.140 | 31 | 0.065 | 0.290 | 0.126 | 72 | 0.361 | 0.146 |
| RRF: hybrid BM25+LaBSE orig+norm | 47 | 0.192 | 0.511 | 0.287 | 31 | 0.194 | 0.484 | 0.277 | 72 | 0.528 | 0.305 |

## Sensitivity: RRF k (seed_all, BM25 orig+norm)

| k | Sinhala Hit@10 | Latin Hit@10 | All Hit@10 |
|---:|---:|---:|---:|
| 10 | 0.368 | 0.500 | 0.463 |
| 60 | 0.316 | 0.528 | 0.439 |
| 100 | 0.316 | 0.528 | 0.439 |
