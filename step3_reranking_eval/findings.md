# Step 3: whole-item vs partial-exposure profiles in an offline re-ranking eval

Question: if a user's history records how much of each article they actually saw, instead of treating every click as fully read, does a simple profile rank their held-out clicks better?

## Setup
- **Data.** EB-NeRD validation split. Profiles come from each user's 3-week history; every front-page impression in the split is re-ranked. Small: 244,647 impressions, 15,342 users. Large (category arms only): 12.6M impressions, 792k users.
- **One relevance model for every arm:** score = ⟨user profile, candidate⟩. AUC, MRR and nDCG@5 per impression, averaged. 95% CIs from a user-clustered bootstrap of paired differences.
- **Exposure estimate per history read** (fraction of text seen, top-down): `scroll` = scroll % / 100; `readtime` = read time × 230 wpm / article words; `scroll+readtime` = the minimum of the two. Missing signal counts as fully read.
- **Units (the 2×2).** Category: profile is a histogram of categories, each read weighted 1 (whole) or by its exposure fraction (partial). Semantic: paragraphs embedded with `intfloat/multilingual-e5-base`; an article is the word-weighted sum of its paragraph embeddings; the partial profile sums only the paragraphs above the exposure cut-off.
- **Decomposition arms** (semantic): `S_whole_engagement_wt` weights the whole article by its exposure fraction (depth only, no content selection); `S_partial_content` uses the seen paragraphs renormalised to unit weight (content only, no depth weighting); `S_unseen` builds the profile from the parts that were *not* seen.

Scripts: `embed.py` (embeddings, CPU), `eval.py` (all arms), `eval_category_full.py` (memory-lean category arms for the large set). Summaries: `summary_small_validation.json`, `summary_small_validation_uncentered.json`, `summary_category_full_validation.json`.

## Results (validation AUC, scroll+readtime exposure)

| | whole item | partial exposure | Δ (95% CI) |
|---|---|---|---|
| category, small | 0.559 | 0.566 | +0.0063 (0.0054 to 0.0073) |
| category, large | 0.560 | 0.567 | +0.0066 (0.0065 to 0.0067) |
| semantic (e5), small | 0.592 | 0.600 | +0.0084 (0.0077 to 0.0092) |

Random = 0.500. MRR and nDCG@5 move the same way.

**Where the semantic gain comes from** (Δ AUC vs `S_whole`, small):
- depth weighting only (`S_whole_engagement_wt`): +0.0074
- seen-paragraph content only (`S_partial_content`): +0.0034
- both (`S_partial`): +0.0084, so content selection adds about +0.001 on top of depth weighting
- profile from the **unseen** parts (`S_unseen`): −0.0082

**Which signal.** Read time alone is as good as scroll+readtime (0.600 vs 0.600 semantic; 0.566 vs 0.566 category). Scroll alone is weaker (0.595 semantic, 0.561 category), consistent with step 1's finding that scroll overstates reading.

**By user.** The gain is largest for users who usually stop early (lowest tercile of mean exposure: semantic 0.602 → 0.612, category 0.565 → 0.574) and smallest for users who usually read to the end (0.579 → 0.585, 0.551 → 0.554).

## What this means
1. Partial exposure helps in every cell of the 2×2, and the category result replicates on the large set with tight CIs. The effect is small in absolute terms, as expected for a dot-product profile model.
2. Most of the gain is "count a read by how much of it was read", not "which parts were read". That is useful but close to dwell-time weighting, which is known.
3. Profiling users by what they skipped does worse than whole-item profiles. So in this form, there is no evidence that users' next clicks fill the gaps of what they partly read. This does not test whether *recommending* gap-filling content would help users; logged clicks on front-page impressions can't show that.

## Caveats
- Semantic arms ran on EB-NeRD small only. The large set has millions of history paragraphs and needs a GPU to embed.
- e5 embeddings must be mean-centred: without subtracting the corpus mean, profiles collapse to the common direction and every semantic arm scores about 0.50–0.51 AUC (`summary_small_validation_uncentered.json`; `CENTER=0` in `eval.py`).
- The paragraph cut-off maps exposure fraction to word share top-down; it ignores images and layout, which step 1 shows affect scroll.
- Not yet compared against a trained model (e.g. NRMS) with partial-exposure inputs.
