# EAR: Exposure-Aware Ranking

News recommenders usually represent a user by the items they clicked, treating each click as full consumption. EAR models what a reader was actually exposed to within each item, using logged scroll depth and read time, and uses that partial-exposure history to build user profiles for ranking.

This repository contains the analyses and evaluation code on [EB-NeRD](https://recsys.eb.dk/), the Ekstra Bladet news recommendation dataset.

## Analyses

| | question | data | main result |
|---|---|---|---|
| [01 · Exposure signal](analyses/01_exposure_signal) | How much do scroll % and read time say about within-article exposure? | small, large | 36% of reads stop before 50% scroll; scroll exceeds read-time-implied exposure and depends on length and layout |
| [02 · Unread-content test](analyses/02_unread_content_test) | Is the next click closer to the unread part of the previous article? | small | No difference from a full-reader placebo (all p > 0.38) |
| [03 · Exposure-aware re-ranking](analyses/03_reranking) | Do partial-exposure profiles rank held-out clicks better than whole-item profiles? | small, large | Yes in all four cells of the 2×2; semantic AUC 0.592 → 0.600 |

### Re-ranking results (validation AUC)

| | whole item | partial exposure | Δ AUC (95% CI) |
|---|---|---|---|
| category, small | 0.559 | 0.566 | +0.0063 (+0.0054, +0.0073) |
| category, large | 0.560 | 0.567 | +0.0066 (+0.0065, +0.0067) |
| semantic (e5-base), small | 0.592 | 0.600 | +0.0084 (+0.0077, +0.0092) |

Exposure estimated from scroll and read time; same dot-product relevance model in every arm. Details, ablations and per-user breakdowns are in [analyses/03_reranking](analyses/03_reranking).

## Repository layout
```
data/download_ebnerd.sh            downloads EB-NeRD into data/ebnerd_{small,large}/
analyses/01_exposure_signal/       scroll and read-time descriptives, TF-IDF next-click test
analyses/02_unread_content_test/   next-click test with multilingual-e5-base embeddings
analyses/03_reranking/             whole vs partial exposure × category vs semantic re-ranking
```

## Setup
```bash
pip install -r requirements.txt
bash data/download_ebnerd.sh small    # 84 MB
bash data/download_ebnerd.sh large    # 3.2 GB
```
Each dataset folder contains `articles.parquet` and `train/`, `validation/` with `history.parquet` and `behaviors.parquet`.

## Reproducing the results
```bash
S=data/ebnerd_small; L=data/ebnerd_large

python analyses/01_exposure_signal/exposure_signal.py       $S out/01
python analyses/01_exposure_signal/descriptives_large.py    $L out/01_large
python analyses/02_unread_content_test/unread_content_test.py $S out/02

python analyses/03_reranking/embed.py        $S validation out/emb_small_validation.npz
python analyses/03_reranking/rerank_eval.py  $S validation out/03 out/emb_small_validation.npz
python analyses/03_reranking/rerank_eval_category_large.py $L validation out/03_large
```
All reported runs used CPU. Semantic arms are reported on EB-NeRD small; embedding the paragraph histories of EB-NeRD large is practical only on a GPU.

## License
Apache 2.0. EB-NeRD is distributed separately under its own terms and is not included here.
