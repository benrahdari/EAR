# Step 1b: unread-part test with dense embeddings (e5-base)

Data: EB-NeRD *small* (train + validation), same selection as step 1: 117k front-page impressions right after an article read in the same session, previous article ≥6 paragraphs and ≥200 words. Model: `intfloat/multilingual-e5-base` ("query: " prefix on all texts, max 512 tokens, normalized), CPU. Embedded 40,276 paragraphs of the previous articles and 3,394 candidate articles (title + subtitle + body). Script: `step1b.py`; numbers: `results.json`; per-pair rows: `followup_pairs.csv`.

## Short answer
The null holds. With e5-base, the clicked article is no closer to the unread part of the previous article than to the read part, relative to the other articles shown, and partial readers look the same as the full-reader placebo in every scroll bin (all p > 0.38). This holds for mean-pooled parts and for best-matching paragraph.

More importantly, a sanity check shows the test cannot detect the effect in this setting: **the clicked article is not closer to the previous article as a whole than the other candidates are.** So the next click from the front page carries almost no topical link to the article just read, whatever text representation is used.

## Numbers
Unseen-minus-seen similarity, clicked minus mean of other candidates (partial readers minus placebo; e5 cosine units, SD of the per-pair score ≈ 0.008):

| split bin (scroll %) | partial | placebo | diff (se) | MW p |
|---|---|---|---|---|
| 5–25 | −0.00036 | −0.00045 | +0.00009 (0.00008) | 0.62 |
| 25–50 | −0.00029 | −0.00034 | +0.00005 (0.00006) | 0.55 |
| 50–75 | −0.00018 | −0.00015 | −0.00003 (0.00007) | 0.95 |
| 75–99 | −0.00010 | −0.00007 | −0.00004 (0.00010) | 0.53 |

Both groups lean slightly toward the *seen* (opening) part, as with TF-IDF in step 1; the opening carries the headline topic.

Sanity check (30k-impression sample): where does the clicked article rank among in-view candidates by similarity to the previous article (0.5 = chance)?
- e5-base: 0.489 (se 0.002)
- TF-IDF: 0.472 (se 0.002)
- same category as previous article: clicked 23.9% vs 19.6% of candidates, so a small category effect only.

## What this means
The step-1 null was not caused by TF-IDF. In EB-NeRD's logged front-page impressions, what gets clicked next is driven by things other than content continuity with the last article (position, recency, headline appeal, the front-page mix), so comparing seen vs unseen parts has nothing to work with. This test is not a test of the partial-exposure idea; it is a test of a premise (next click follows the last article's content) that does not hold here.

Options considered for the next phase (option 2 was run; see `../step3_reranking_eval`):
1. Use the user's longer reading history as the "information state" rather than only the last article, and first check that content similarity to history predicts clicks at all (positive control).
2. Move to the offline re-ranking evaluation from the plan (step 3): build whole vs partial exposure user profiles and compare how well they rank held-out clicks, which tests usefulness rather than readers' spontaneous behavior.
3. Run the same e5 pipeline on the full dataset (cheap to adapt, but on current evidence likely to repeat the null with tighter error bars).
