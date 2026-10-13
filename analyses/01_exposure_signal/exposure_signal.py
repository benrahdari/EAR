"""Within-article exposure signal in EB-NeRD: scroll percentage and read time, and a TF-IDF test of whether the
next click relates to the unread part of the previous article.

usage: python exposure_signal.py DATA_DIR OUT_DIR
"""
import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

D, OUT = sys.argv[1], sys.argv[2]
R = {}
arts = pd.read_parquet(f"{D}/articles.parquet")
arts["paras"] = arts.body.fillna("").str.split("\n").apply(lambda p: [x for x in p if x.strip()])
arts["n_words"] = arts.body.fillna("").str.split().str.len()
arts["n_paras"] = arts.paras.str.len()
arts["n_imgs"] = arts.image_ids.apply(lambda x: 0 if x is None else len(x))
A = arts.set_index("article_id")

# ---------- A. history reads ----------
h = pd.concat([pd.read_parquet(f"{D}/{s}/history.parquet") for s in ("train", "validation")]).drop_duplicates("user_id")
e = h.explode(["impression_time_fixed", "scroll_percentage_fixed", "article_id_fixed", "read_time_fixed"])
e.columns = ["user", "t", "scroll", "aid", "rt"]
e = e.astype({"scroll": float, "rt": float, "aid": int}).join(A[["n_words", "n_paras", "n_imgs", "premium", "article_type"]], on="aid")
e = e[e.n_words > 0]
R["history_reads"] = len(e)
R["scroll_missing"] = e.scroll.isna().mean()
s = e.dropna(subset=["scroll"])
R["scroll_eq_100"] = (s.scroll == 100).mean()
R["scroll_lt_50"] = (s.scroll < 50).mean()

# implied fraction read from read time at 230 wpm
s = s.assign(frac_rt=np.clip(s.rt / 60 * 230 / s.n_words, 0, 1), frac_sc=s.scroll / 100)
R["spearman_scroll_readtime"] = stats.spearmanr(s.scroll, s.rt).statistic
R["spearman_scroll_impliedfrac"] = stats.spearmanr(s.frac_sc, s.frac_rt).statistic
s["len_bin"] = pd.qcut(s.n_words, 5)
R["by_length"] = s.groupby("len_bin", observed=True).agg(
    median_scroll=("scroll", "median"), share_100=("scroll", lambda x: (x == 100).mean()),
    median_rt=("rt", "median"), median_frac_rt=("frac_rt", "median")).round(3).reset_index().astype(str).to_dict("records")
s["img_bin"] = pd.cut(s.n_imgs, [-1, 0, 1, 3, 100], labels=["0", "1", "2-3", "4+"])
R["by_images"] = s.groupby("img_bin", observed=True).agg(median_scroll=("scroll", "median"), share_100=("scroll", lambda x: (x == 100).mean()), n=("scroll", "size")).round(3).reset_index().astype(str).to_dict("records")
# scroll without reading: reached bottom in < 10 s on articles of 300+ words
long_ = s[s.n_words >= 300]
R["long_articles_scroll100_under10s"] = ((long_.scroll == 100) & (long_.rt < 10)).mean()
R["long_articles_scroll_lt50_over60s"] = ((long_.scroll < 50) & (long_.rt > 60)).mean()
# user habit: share of variance in scroll explained by user
uc = s.groupby("user").scroll.transform("count")
ss = s[uc >= 20]
R["user_share_of_scroll_variance"] = ss.groupby("user").scroll.mean().var() / ss.scroll.var()
# same user, same article read twice: test-retest
rep = s.groupby(["user", "aid"]).filter(lambda x: len(x) >= 2) if False else None

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(s.scroll, bins=50, color="#4C72B0"); ax[0].set_title("Scroll % per article read"); ax[0].set_xlabel("scroll %")
bl = s.groupby("len_bin", observed=True)
ax[1].plot(range(5), bl.scroll.median(), "o-", label="median scroll %")
ax[1].plot(range(5), bl.scroll.apply(lambda x: (x == 100).mean() * 100), "s-", label="% reads at 100")
ax[1].set_xticks(range(5), [f"{int(i.left)}-{int(i.right)}" for i in bl.groups.keys()], rotation=20)
ax[1].set_xlabel("article length (words, quintiles)"); ax[1].legend(); ax[1].set_title("Scroll vs. length")
hb = ax[2].hexbin(s.frac_sc, s.frac_rt, gridsize=30, bins="log", cmap="Blues")
ax[2].set_xlabel("scroll fraction"); ax[2].set_ylabel("read-time-implied fraction (230 wpm)"); ax[2].set_title("Two estimates of fraction read")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_scroll_profile.png", dpi=130)

# ---------- B. does what was NOT seen predict what is clicked next? ----------
b = pd.concat([pd.read_parquet(f"{D}/{s_}/behaviors.parquet") for s_ in ("train", "validation")]).sort_values(["user_id", "impression_time"])
b["clk"] = b.article_ids_clicked.str[0]
g = b.groupby("user_id")
for c, src in [("prevA", "clk"), ("prevS", "next_scroll_percentage"), ("prevR", "next_read_time"), ("prevSess", "session_id")]:
    b[c] = g[src].shift()
f = b[b.article_id.isna() & (b.prevSess == b.session_id) & b.prevS.notna() & (b.article_ids_clicked.str.len() == 1)].copy()
f["prevA"] = f.prevA.astype(int)
f = f[f.prevA.isin(A.index)]
f = f.join(A[["paras", "n_paras", "n_words"]], on="prevA")
f = f[(f.n_paras >= 6) & (f.n_words >= 200)]
R["sessions_followup_impressions"] = len(f)

texts = (arts.title.fillna("") + " " + arts.subtitle.fillna("") + " " + arts.body.fillna("")).values
vec = TfidfVectorizer(min_df=3, max_df=0.3, sublinear_tf=True).fit(texts)
idx = {a: i for i, a in enumerate(arts.article_id)}
M = vec.transform(texts)

def split(paras, frac):
    w = np.cumsum([len(p.split()) for p in paras]); w = w / w[-1]
    k = max(1, min(len(paras) - 1, int(np.searchsorted(w, frac, side="right"))))
    return " ".join(paras[:k]), " ".join(paras[k:])

BINS = [(5, 25), (25, 50), (50, 75), (75, 99.99)]
rows = []
for r in f.itertuples():
    cands = [c for c in r.article_ids_inview if c in idx and c != r.prevA]
    if r.clk not in cands or len(cands) < 3 or r.prevS <= 5:
        continue
    full = r.prevS >= 100
    # partial readers: split at their scroll point; full readers (placebo): split at every bin midpoint
    splits = [((lo + hi) / 2 / 100, (lo, hi)) for lo, hi in BINS] if full else \
             [(r.prevS / 100, next(b_ for b_ in BINS if b_[0] < r.prevS <= b_[1]))]
    C = M[[idx[c] for c in cands]]; ci = cands.index(r.clk)
    for frac, b_ in splits:
        V = vec.transform(list(split(r.paras, frac)))
        S = (C @ V.T).toarray(); diff = S[:, 1] - S[:, 0]
        rows.append(dict(user=r.user_id, scroll=r.prevS, rt=r.prevR, full=full, bin=f"{b_[0]}-{int(b_[1])}",
                         delta=diff[ci] - np.delete(diff, ci).mean(),
                         rank=(diff[ci] > np.delete(diff, ci)).mean()))
P = pd.DataFrame(rows)
P.to_csv(f"{OUT}/followup_pairs.csv", index=False)
P["bin"] = pd.Categorical(P.bin, [f"{lo}-{int(hi)}" for lo, hi in BINS], ordered=True)
T = P.groupby(["bin", "full"], observed=True).agg(mean=("delta", "mean"), sem=("delta", "sem"), rank=("rank", "mean"), n=("delta", "size")).reset_index()
R["delta_partial_vs_placebo"] = T.round(6).astype(str).to_dict("records")
dd = {}
for bn, x in P.groupby("bin", observed=True):
    a, c = x[~x.full].delta, x[x.full].delta
    dd[bn] = dict(diff=float(a.mean() - c.mean()), se=float(np.sqrt(a.var() / len(a) + c.var() / len(c))),
                  mannwhitney_p=float(stats.mannwhitneyu(a, c).pvalue))
R["did_partial_minus_placebo"] = dd

fig, ax = plt.subplots(figsize=(6.5, 4))
for k, (full, lab) in enumerate([(False, "stopped here (partial read)"), (True, "read to 100% (placebo, same split)")]):
    x = T[T.full == full]
    ax.errorbar(np.arange(len(x)) + (k - .5) * .1, x["mean"], yerr=1.96 * x["sem"], fmt="o-", capsize=4, label=lab)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(range(len(BINS)), [f"{lo}-{int(hi)}" for lo, hi in BINS])
ax.set_xlabel("split point in previous article (scroll %)"); ax.set_ylabel("clicked − other candidates")
ax.set_title("Similarity to unseen part minus similarity to seen part"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_unseen_pull.png", dpi=130)
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
