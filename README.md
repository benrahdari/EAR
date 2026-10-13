# EAR: Exposure-Aware Ranking

Recommendation from partial information exposure: first analyses on EB-NeRD.

Most recommenders treat a clicked article as fully consumed. The idea here is to model what a reader was actually exposed to inside an article (the parts they saw) and use that partial history for recommendation. This repo holds the first round of analyses on [EB-NeRD](https://recsys.eb.dk/), starting from the question of whether its scroll percentage and read time carry enough signal to tell which parts of an article were read.

Status: early, exploratory. Code, numbers and figures are here; the dataset is not (see [Data](#data)).

## Summary of findings

**1. Is there enough signal? Coarse yes, paragraph-level no** ([step1_ebnerd_signal](step1_ebnerd_signal/findings.md)).
- 36% of reads stop before 50% scroll and 40% reach 100%, so "clicked = read" is wrong for at least a third of reads.
- Scroll overstates reading. Scroll and read time correlate only moderately (Spearman 0.52). At 230 wpm, read time implies a median of 13–39% of the text read while median scroll is 65–98%.
- Scroll depends on the page: short articles and articles without images hit 100% far more often. User identity explains about 16% of scroll variance.
- All of these hold on the full dataset (975k users, 132M reads) to within a point of the small set.
- Takeaway: exposure should be modelled probabilistically from scroll *and* read time with length/layout covariates, not as a deterministic "read up to scroll %" rule.

**2. Do readers' next clicks follow the unread part of what they just read? No detectable effect** ([step1b_embedding_test](step1b_embedding_test/findings.md)).
- With TF-IDF and with multilingual-e5-base paragraph embeddings, partial readers look the same as a full-reader placebo in every scroll bin (all p > 0.38).
- The reason is upstream: the next front-page click is no closer to the last-read article than the other candidates shown (rank 0.49 with e5; chance 0.50). So this test checks a premise that doesn't hold in EB-NeRD's logs. It is not evidence against partial exposure.

**3. Do partial-exposure profiles rank held-out clicks better? Yes, mostly through read-depth weighting** ([step3_reranking_eval](step3_reranking_eval/findings.md)).

| validation AUC | whole item | partial exposure |
|---|---|---|
| category, small | 0.559 | 0.566 |
| category, large | 0.560 | 0.567 |
| semantic (e5), small | 0.592 | 0.600 |

- Partial exposure wins in all four cells of the whole/partial × category/semantic comparison, with the same simple dot-product relevance model. Gains are largest for users who usually stop early.
- Of the +0.0084 semantic gain, +0.0074 comes from weighting each read by how much was read. Knowing *which* paragraphs were seen adds about +0.001.
- A profile built from the parts users skipped does worse (−0.008), so in this form there is no sign that clicks fill gaps in partly-read articles.
- Read time alone is as useful as scroll + read time; scroll alone is weaker.

(Step 2 of the original plan, mapping scroll position to paragraphs, is implemented inside `step3_reranking_eval/eval.py`.)

## Open questions
- Is finer exposure data logged anywhere (viewport positions, scroll events over time, time per screen)? Scroll % and total read time are the bottleneck for paragraph-level exposure.
- How large are the gains on top of a trained model such as NRMS, rather than a dot-product profile?
- The information-deficit hypothesis is about what *recommending* gap-filling content does, which logged front-page clicks can't show. What would be a credible offline or online test?

## Repo layout
```
data/download_ebnerd.sh        fetches EB-NeRD into data/ebnerd_{small,large}/
step1_ebnerd_signal/           signal check: step1.py (small), desc_full.py (full descriptives), findings, figures
step1b_embedding_test/         unread-part test with e5-base: step1b.py, findings, figure
step3_reranking_eval/          2x2 re-ranking eval: embed.py, eval.py, eval_category_full.py, findings, summaries
```

## Data
EB-NeRD is not redistributed here. To download it:
```bash
bash data/download_ebnerd.sh small          # 84 MB
bash data/download_ebnerd.sh large          # 3.2 GB
```
Each folder ends up with `articles.parquet`, `train/` and `validation/` (`history.parquet`, `behaviors.parquet`).

## Reproducing
```bash
pip install -r requirements.txt
D=data/ebnerd_small

python step1_ebnerd_signal/step1.py      $D out/step1
python step1_ebnerd_signal/desc_full.py  data/ebnerd_large out/step1_full
python step1b_embedding_test/step1b.py   $D out/step1b             # embeds ~44k texts with e5-base

python step3_reranking_eval/embed.py     $D validation out/emb_small_validation.npz
python step3_reranking_eval/eval.py      $D validation out/step3 out/emb_small_validation.npz
CENTER=0 python step3_reranking_eval/eval.py $D validation out/step3_uncentered out/emb_small_validation.npz
python step3_reranking_eval/eval_category_full.py data/ebnerd_large validation out/step3_full
```
All runs used CPU only. The committed summaries were renamed with a dataset prefix (e.g. `summary_small_validation.json`). The step 3 semantic arms have only been run on the small set; embedding the large set's history paragraphs needs a GPU.
