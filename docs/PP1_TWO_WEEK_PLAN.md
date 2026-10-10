# Two-Week Progress Presentation 1 Plan

Goal: show a working Singlish query → real YouTube evidence retrieval interface, report exploratory baseline diagnostics without exaggerating them, and explain the methodology and upcoming experiments.

| Days | Work | Deliverable |
|---|---|---|
| 1–2 | Run the bundled Gradio app, pick reliable demo queries and failure cases, validate dataset profile | Runnable local demo and 2 screenshots |
| 3–4 | Improve YouTube comment collector to preserve comment IDs, reply parent IDs, timestamps; keep API keys out of code | Improved collection script + new traceable sample |
| 5–6 | Create a pilot annotation guideline, sample 30–50 queries across intended domains, select both relevant/irrelevant candidate comments | Independently reviewed pilot labels (at least two bilingual readers if available) |
| 7–9 | Download/build optional LaBSE index, compare same queries with BM25 and TF-IDF; collect qualitative successes/failures | Cross-script semantic retrieval proof of concept (if compute allows) |
| 10–11 | Implement small rule-based normalization proof of concept or begin structured spelling-pair creation; log failures | Ablation outline and example cases |
| 12 | Prepare architecture, dataset and results slides, diagram distinguishing current and planned modules | Draft 7–9-slide presentation |
| 13 | Rehearse demo on presentation machine, ensure offline mode and fallback screenshot | Recorded working demo and rehearsal notes |
| 14 | Final presentation review, source/metrics audit, rehearse questions | PP1 package |

## Presentation outline (8 slides)

1. **Title / individual component boundary** — Explainable RAG Knowledge Assistant & Singlish Retrieval Engine (Component 3).
2. **Problem and research gap** — Romanized Sinhala (Singlish) questions vs Sinhala-script evidence; noisy mixed-script YouTube comments.
3. **Specific objectives and research hypotheses** — data benchmark, normalization + dense retrieval, thread-aware RAG, claim verification and UI.
4. **Data collection** — 11,176 source rows; cleaning produces 10,343 distinct video/text records over 48 valid video IDs; 366 query–comment seed pairs; 128 exact source text matches; 82 unique linked queries. Explain missing comment IDs and incomplete domain metadata.
5. **Implemented architecture (solid) vs planned architecture (dashed)** — CSV ingestion → cleaning → BM25/TF-IDF → results UI (implemented); independent judgment → LaBSE/normalization → RAG → claim verification (planned).
6. **Live demonstration** — enter a Singlish query; return ranked real comments and video links; show both a successful lexical query and a cross-script failure. Make clear video links are not individual comment permalinks.
7. **Preliminary results and limitations** — over 82 seed-linked unique queries: BM25 known-positive Hit@10 = 23.17%; char TF-IDF = 25.61%. Clarify that seed pairs are incomplete, may be query-from-document, and NOT independent nDCG/Recall or model validation.
8. **Next sprint and technical risks** — ID-aware collection, two-annotator graded relevance, independent evaluation, multilingual dense embeddings and ablation; then thread-aware RAG and evidence verification.

## Live demo script (approximately 90 seconds)

- "This is my implemented baseline search system over a corpus of real YouTube comments collected for the research."
- Enter a known working example such as `Bandi full magic mix eka kade ganna thiyenawada?` and choose BM25.
- "The interface ranks the corpus and shows original comments and source videos. The scores are ranking scores, not verified relevance labels."
- Switch to Character TF-IDF and show how ordering differs.
- Show an example of a cross-script query where BM25 fails or ranks poorly. "This motivates the planned Singlish normalization and multilingual dense retrieval stage; its impact has not yet been established."
- Open the baseline diagnostics tab. "These results are a diagnostic on 82 queries with known positive pairs; we still need independent graded annotation for final nDCG@10, Recall@50, and statistical tests."

## Likely examiner questions

**Q: What is novel?** The *hypothesis* is that Singlish normalization and multi-view semantic retrieval will improve cross-script retrieval over lexical and zero-shot baselines, with subsequent thread-aware RAG and claim verification. This hypothesis is not yet proven.

**Q: Where are the negatives?** They are not independently judged yet. The seed dataset contains positive examples and will be expanded through pooling and blinded annotation.

**Q: Why do you need comment IDs?** To reliably cite individual evidence and build threads; the old collector kept only video IDs.

**Q: Has RAG been implemented?** No. The PP1 implementation demonstrates retrieval baselines and data preparation. RAG and NLI are planned follow-on stages.

**Q: Why isn't 25.61% a final score?** It is known-positive Hit@10 over exact-linked seed pairs, not recall over fully judged documents or a held-out test set. It is a useful pilot diagnostic only.

## Priority if time is very tight

If only a few days remain, prioritize: **(1)** reliable UI, **(2)** real-source examples, **(3)** honest measured baseline table, **(4)** future experimentation plan. Do not spend the entire remaining time on expensive model fine-tuning or polishing a fake end-to-end chatbot.
