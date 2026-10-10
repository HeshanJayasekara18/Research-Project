# Exploratory cross-script diagnostic (NOT a benchmark)

Exploratory: unverified known-positive seed pairs, incomplete qrels, queries may be written from the documents; variants are synthetic. NOT nDCG/Recall on an independent benchmark.

| Method | Category | n | Hit@1 | Hit@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| BM25 (lexical) | singlish_to_latin | 36 | 0.222 | 0.500 | 0.332 |
| BM25 (lexical) | singlish_to_sinhala | 57 | 0.018 | 0.018 | 0.018 |
| BM25 (lexical) | spelling_variants | 135 | 0.096 | 0.215 | 0.131 |
| BM25 (lexical) | all_original_queries | 82 | 0.110 | 0.232 | 0.158 |
| Character TF-IDF | singlish_to_latin | 36 | 0.306 | 0.583 | 0.401 |
| Character TF-IDF | singlish_to_sinhala | 57 | 0.000 | 0.000 | 0.000 |
| Character TF-IDF | spelling_variants | 135 | 0.118 | 0.222 | 0.151 |
| Character TF-IDF | all_original_queries | 82 | 0.134 | 0.256 | 0.176 |
| LaBSE (semantic) | singlish_to_latin | 36 | 0.056 | 0.167 | 0.086 |
| LaBSE (semantic) | singlish_to_sinhala | 57 | 0.000 | 0.053 | 0.009 |
| LaBSE (semantic) | spelling_variants | 135 | 0.015 | 0.082 | 0.027 |
| LaBSE (semantic) | all_original_queries | 82 | 0.024 | 0.110 | 0.044 |
