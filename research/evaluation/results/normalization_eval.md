# Milestone 2 - Normalisation evaluation (EXPLORATORY, not a benchmark)

Reference words are AI-drafted and await native-speaker review; retrieval positives are unverified seed pairs with incomplete qrels. Hit@k/MRR@10 are exploratory; no significance testing; n is small.

## A. Word-level normalisation quality (draft references)

| Category | n | Correct | Wrong (silent) | Wrong (flagged ambiguous) | Abstained (kept Latin) | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 99 | 94 | 2 | 2 | 1 | 0.950 |
| content | 57 | 54 | 0 | 2 | 1 | 0.947 |
| english | 10 | 10 | 0 | 0 | 0 | 1.000 |
| function | 28 | 27 | 1 | 0 | 0 | 0.964 |
| named_entity | 4 | 3 | 1 | 0 | 0 | 0.750 |

## B. Retrieval: original vs normalised query

Queries: 82; query-level normalisation status: {'normalized': 53, 'partial': 29}; mean word coverage 0.77

| Method | Category | n | Hit@1 | Hit@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| BM25 (lexical) | all_original_queries | 82 | 0.110 | 0.232 | 0.158 |
| BM25 (lexical) | singlish_to_latin | 36 | 0.222 | 0.500 | 0.332 |
| BM25 (lexical) | singlish_to_sinhala | 57 | 0.018 | 0.018 | 0.018 |
| BM25 (lexical) | singlish_to_sinhala|norm_normalized | 42 | 0.000 | 0.000 | 0.000 |
| BM25 (lexical) | singlish_to_sinhala|norm_partial | 15 | 0.067 | 0.067 | 0.067 |
| Normalized query -> BM25 (experimental) | all_original_queries | 82 | 0.146 | 0.390 | 0.222 |
| Normalized query -> BM25 (experimental) | singlish_to_latin | 36 | 0.083 | 0.194 | 0.119 |
| Normalized query -> BM25 (experimental) | singlish_to_sinhala | 57 | 0.158 | 0.439 | 0.244 |
| Normalized query -> BM25 (experimental) | singlish_to_sinhala|norm_normalized | 42 | 0.143 | 0.452 | 0.237 |
| Normalized query -> BM25 (experimental) | singlish_to_sinhala|norm_partial | 15 | 0.200 | 0.400 | 0.263 |
| LaBSE (semantic) | all_original_queries | 82 | 0.024 | 0.110 | 0.044 |
| LaBSE (semantic) | singlish_to_latin | 36 | 0.056 | 0.167 | 0.086 |
| LaBSE (semantic) | singlish_to_sinhala | 57 | 0.000 | 0.053 | 0.009 |
| LaBSE (semantic) | singlish_to_sinhala|norm_normalized | 42 | 0.000 | 0.048 | 0.007 |
| LaBSE (semantic) | singlish_to_sinhala|norm_partial | 15 | 0.000 | 0.067 | 0.013 |
| Normalized query -> LaBSE (experimental) | all_original_queries | 82 | 0.134 | 0.378 | 0.207 |
| Normalized query -> LaBSE (experimental) | singlish_to_latin | 36 | 0.056 | 0.194 | 0.091 |
| Normalized query -> LaBSE (experimental) | singlish_to_sinhala | 57 | 0.158 | 0.421 | 0.241 |
| Normalized query -> LaBSE (experimental) | singlish_to_sinhala|norm_normalized | 42 | 0.167 | 0.286 | 0.211 |
| Normalized query -> LaBSE (experimental) | singlish_to_sinhala|norm_partial | 15 | 0.133 | 0.800 | 0.325 |

### Leakage ablation (lexicon excludes all known-positive comments)

| Method | Category | n | Hit@1 | Hit@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| Normalized (lexicon excl. known positives) -> BM25 (lexical) | singlish_to_latin | 36 | 0.083 | 0.222 | 0.122 |
| Normalized (lexicon excl. known positives) -> BM25 (lexical) | singlish_to_sinhala | 57 | 0.158 | 0.368 | 0.223 |
| Normalized (lexicon excl. known positives) -> LaBSE (semantic) | singlish_to_latin | 36 | 0.056 | 0.167 | 0.088 |
| Normalized (lexicon excl. known positives) -> LaBSE (semantic) | singlish_to_sinhala | 57 | 0.123 | 0.456 | 0.210 |
