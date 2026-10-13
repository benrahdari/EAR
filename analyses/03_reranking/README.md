# 03 · Exposure-aware re-ranking

Compares user profiles built from whole-item histories with profiles built from estimated partial exposure, at two granularities (article category and paragraph-level semantics), under a single relevance model.

## Setup
- **Data:** EB-NeRD validation split. Profiles use each user's 3-week history; every front-page impression in the split is re-ranked.
  - small: 244,647 impressions, 15,342 users (all arms)
  - large: 12,566,385 impressions, 791,582 users (category arms)
- **Relevance model:** score = ⟨user profile, candidate⟩ for every arm. Metrics per impression: AUC, MRR, nDCG@5. 95% confidence intervals from a user-clustered bootstrap (1,000 resamples) of paired differences.
- **Exposure estimate** per history read, as the fraction of the article seen from the top:
  - `scroll`: scroll % / 100
  - `readtime`: read time × 230 wpm / word count
  - `scroll+readtime`: minimum of the two
  - reads with no signal count as fully read
- **Profiles**

| arm | description |
|---|---|
| `C_whole` | category histogram, one count per read |
| `C_partial` | category histogram, each read weighted by its exposure fraction |
| `S_whole` | sum of article embeddings (word-weighted mean of paragraph embeddings) |
| `S_partial` | sum over reads of the paragraphs within the exposed fraction |
| `S_whole_engagement_wt` | whole-article embedding weighted by exposure fraction |
| `S_partial_content` | exposed paragraphs only, renormalised to unit weight per read |
| `S_unseen` | paragraphs beyond the exposed fraction, renormalised |

Paragraph and candidate embeddings use `intfloat/multilingual-e5-base` and are mean-centred over the corpus before scoring.

## Results

**Validation AUC, `scroll+readtime` exposure**

| | whole item | partial exposure | Δ AUC (95% CI) |
|---|---|---|---|
| category, small | 0.5593 | 0.5657 | +0.0063 (+0.0054, +0.0073) |
| category, large | 0.5600 | 0.5666 | +0.0066 (+0.0065, +0.0067) |
| semantic, small | 0.5915 | 0.5999 | +0.0084 (+0.0077, +0.0092) |

Random ranking: 0.500. MRR and nDCG@5 follow the same ordering (`summary_*.json`).

**Decomposition of the semantic gain** (Δ AUC vs `S_whole`, small, `scroll+readtime`)

| arm | Δ AUC (95% CI) |
|---|---|
| `S_partial` | +0.0084 (+0.0077, +0.0092) |
| `S_whole_engagement_wt` | +0.0074 (+0.0068, +0.0081) |
| `S_partial_content` | +0.0034 (+0.0027, +0.0042) |
| `S_unseen` | −0.0082 (−0.0089, −0.0076) |

**Exposure signal** (AUC, small)

| | scroll | readtime | scroll+readtime |
|---|---|---|---|
| `C_partial` | 0.5613 | 0.5658 | 0.5657 |
| `S_partial` | 0.5952 | 0.6000 | 0.5999 |

**By user reading depth** (AUC, small; terciles of the user's mean scroll exposure)

| tercile | `C_whole` | `C_partial` | `S_whole` | `S_partial` |
|---|---|---|---|---|
| low | 0.5646 | 0.5741 | 0.6019 | 0.6123 |
| mid | 0.5628 | 0.5692 | 0.5934 | 0.6028 |
| high | 0.5506 | 0.5538 | 0.5793 | 0.5849 |

Summary:
- Partial-exposure profiles outperform whole-item profiles in all four cells; the category result replicates on EB-NeRD large.
- Most of the semantic gain comes from weighting each read by its exposure; selecting the exposed paragraphs adds about +0.001 AUC on top of that.
- Profiles built from unexposed content underperform whole-item profiles.
- Read time alone matches the combined signal; scroll alone is weaker.
- Gains are largest for users who typically stop early.

Without mean-centring, e5 profiles collapse toward the shared embedding direction and every semantic arm scores 0.50–0.51 AUC (`summary_small_validation_uncentered.json`).

## Files
- `embed.py DATA_DIR SPLIT OUT.npz`: paragraph and candidate embeddings
- `rerank_eval.py DATA_DIR SPLIT OUT_DIR [EMB.npz]`: all arms (category only without `EMB.npz`); `CENTER=0` disables mean-centring
- `rerank_eval_category_large.py DATA_DIR SPLIT OUT_DIR`: memory-efficient category arms for EB-NeRD large
- `summary_small_validation.json`, `summary_small_validation_uncentered.json`, `summary_category_full_validation.json`: results reported above
