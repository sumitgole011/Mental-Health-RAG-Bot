# Retrieval change and paired pilot

The Flask app now reads UTF-8 `.txt` documents relative to app.py, filters chunks with fewer than 100 alphabetic characters, and fuses TF-IDF and normalized semantic cosine rankings using reciprocal rank fusion (20 candidates per ranking, constant 60, k=2). Query cleanup removes generic source-reference wording. Chunk size 1000 and overlap 200, MiniLM embeddings, and top-2 output are unchanged. Page fragments are filtered; comprehensive bibliography or clinical-content filtering is not claimed. Dense cosine is exact in-memory search, so this approach is intended for this small corpus, not large indexes. Chroma remains used by the comparison baseline and existing experimental voice script.

The retriever is shared between app and evaluation. Modern LangChain import paths and explicit text ingestion replace the implicit Unstructured loader. TTS and generation behavior are unchanged; no clinical effectiveness or overall answer accuracy is established.

## Reproduce

    pip install -r requirements.txt -r requirements-evaluation.txt
    python evaluation/compare_retrieval.py
    python -m unittest discover -s tests -v

First execution downloads sentence-transformers/all-MiniLM-L6-v2; the comparison requires no Groq key. Run from repository root. Baselines use explicit UTF-8 text ingestion, an adaptation of the old DirectoryLoader behavior. For offline runs, cache the model first. Exact tested environment is in evaluation/requirements-tested.txt.

## Results, 2026-11-18 (India)

| Set | Old Chroma MMR | Old Chroma similarity | Candidate hybrid |
|---|---:|---:|---:|
| Previously exposed 14 questions | 1/14 | 3/14 | 10/14 |
| Fresh 10 questions | 5/10 | 8/10 | 8/10 |
| Fresh MRR@2 | 0.50 | 0.65 | 0.70 |

Scores count whether a top-2 chunk fully contains the annotated short evidence anchor. They are not full-answer coverage, hallucination, clinical quality, or chatbot accuracy. Fresh results are a 30 percentage-point increase over adapted old MMR, and tied in hit count with similarity search. Do not claim superiority over similarity search on this sample. Multiple implementation changes are combined; this is not an ablation isolating any one change.

The 10 fresh questions and candidate implementation were frozen before the first paired run; no candidate tuning followed those results. They cover additional passages in only the wellbeing booklet, not the LEAP source. The 14 exposed questions are development data. All questions/labels were created by the implementing assistant; they are not independent or blinded. The test sets are small, and an alternative valid passage can be scored as a miss. Some questions need more evidence than their short anchor. A larger independently reviewed test set with alternative relevant passages and unsupported queries is needed before generalizing. No live Groq generation or end-to-end browser test was run because no API key was available.

Validation: six repository/retrieval unit tests passed locally. Tests exercise offset preservation, fragment handling, empty corpus behavior, ranking with deterministic vectors and existing source contracts. They do not download a model or invoke an external API. The paired semantic comparison was separately executed using actual cached MiniLM weights.

## Resume wording

If describing this assisted improvement, retain its narrow scope: "Improved top-2 evidence retrieval from 5/10 to 8/10 questions in a small paired pilot by combining lexical and semantic ranking with fragment filtering." This does not describe an independently demonstrated accuracy gain, and should be used only once you understand and can explain the implementation and test design.
