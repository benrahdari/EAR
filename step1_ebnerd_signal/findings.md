# Step 1: Is there enough signal in EB-NeRD to infer partial exposure?

Data: EB-NeRD *small* (train + validation), downloaded from the official S3 bucket. 2.46M history reads with scroll % and read time; 117k front-page impressions that follow an article read in the same session. Script: `step1.py`; numbers: `results.json`.

## Short answer
Yes for a coarse signal, no for paragraph-level reading as is. Scroll % varies a lot across reads, but it measures how far the page moved, not how much text was read. It needs to be combined with read time and corrected for length and layout. That fits the draft's probabilistic examination model P(X | H) better than a deterministic "read up to scroll %" rule.

## What the data shows
- **Plenty of variation.** Scroll is missing for 7% of reads. 40% of reads reach 100%, and 36% stop before 50% (fig1, left). So item-level history ("clicked = read") is wrong for at least a third of reads.
- **Scroll overstates reading.** Scroll and read time agree only moderately (Spearman 0.52). At 230 words/min, read time implies a median of 13–39% of the text read, while median scroll is 65–98% (fig1, right). On articles with 300+ words, 4.9% of reads hit 100% in under 10 seconds.
- **Scroll depends on the page, not just the reader.** Short articles hit 100% about half the time almost automatically (median scroll 98 for <200 words vs 65 for >300). Articles with no images reach 100% in 50% of reads vs 38–40% with images. Scroll is in page-height units, so it needs a length/layout correction before mapping to paragraphs.
- **Readers have habits.** User identity explains about 16% of scroll variance (users with ≥20 reads), so per-user calibration should help.

## A first test of the hypothesis (null)
Question: after a reader stops partway through article A, does the next article they click resemble A's **unseen** part more than its **seen** part, compared with the other articles shown in the same impression?

Method: split A at the scroll point by paragraph word share; TF-IDF cosine of each candidate with the seen and unseen parts; clicked minus mean of other candidates. Placebo: readers who reached 100%, split at the same points.

Result (fig2): no consistent pull toward the unseen part beyond the placebo. Clicked items lean toward the article's opening regardless of where the reader stopped (likely the headline topic). The only significant difference is at 5–25% scroll and it goes the other way.

How to read this: it is a weak test, not evidence against the idea. The candidates shown were not chosen to fill gaps, TF-IDF over whole paragraphs is a crude stand-in for information units, and the draft's claim is that *recommending* gap-filling content helps, not that readers already seek it out. It does say the effect will not show up in raw click logs for free.

## Implications for steps 2–3
1. Model exposure as a probability per paragraph that uses both scroll and read time (e.g. scroll gives an upper bound on what was on screen, read time how much of it was likely read), with article length and image count as covariates.
2. Use finer information units (entities or sentence embeddings) rather than paragraph TF-IDF before running the 2×2 comparison.
3. A cleaner test of the core claim needs counterfactual candidates (offline re-ranking evaluated on held-out clicks) rather than the logged impressions alone.

## Update: descriptive analysis on the full dataset
Script `desc_full.py`, numbers `results_full.json`, figure `fig1_scroll_profile_full.png`. 975k users, 131.9M history reads (118.0M with scroll). Shares and the user-variance figure use every read; medians and correlations use a 2% random sample (2.36M reads).

| | small | full |
|---|---|---|
| scroll missing | 7.1% | 7.0% |
| reads at 100% | 40.3% | 40.2% |
| reads below 50% | 35.8% | 35.8% |
| Spearman scroll vs read time | 0.52 | 0.52 |
| long articles at 100% in <10 s | 4.9% | 4.8% |
| user share of scroll variance | 16% | 16% |

The length and image breakdowns match to within a point. The small set is representative for these statistics; the full set adds nothing new descriptively.
