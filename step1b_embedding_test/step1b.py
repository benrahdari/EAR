"""Step 1b: rerun the 'next click leans toward the unread part' test with dense embeddings
(intfloat/multilingual-e5-base) instead of TF-IDF. Same design as step1_ebnerd_signal/step1.py, part B.

usage: python step1b.py DATA_DIR OUT_DIR [model]
"""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sentence_transformers import SentenceTransformer

D, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "intfloat/multilingual-e5-base"
os.makedirs(OUT, exist_ok=True)
R = {"model": MODEL}

arts = pd.read_parquet(f"{D}/articles.parquet")
arts["paras"] = arts.body.fillna("").str.split("\n").apply(lambda p: [x for x in p if x.strip()])
arts["n_words"] = arts.body.fillna("").str.split().str.len()
arts["n_paras"] = arts.paras.str.len()
A = arts.set_index("article_id")

# ---------- follow-up impressions (identical selection to step 1) ----------
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

# ---------- embeddings ----------
model = SentenceTransformer(MODEL, device="cpu")
model.max_seq_length = 512
enc = lambda xs: model.encode(["query: " + x for x in xs], batch_size=32, normalize_embeddings=True,
                              show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

cand_ids = sorted(set(a for xs in f.article_ids_inview for a in xs if a in A.index))
cache = os.environ.get("EMB_CACHE", f"{OUT}/emb_cache.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); CE, PE, pkeys = z["CE"], z["PE"], list(map(tuple, z["pkeys"]))
else:
    ctext = (A.loc[cand_ids].title.fillna("") + ". " + A.loc[cand_ids].subtitle.fillna("") + " " + A.loc[cand_ids].body.fillna("")).tolist()
    CE = enc(ctext)
    pkeys = [(a, i) for a in sorted(f.prevA.unique()) for i in range(len(A.at[a, "paras"]))]
    PE = enc([A.at[a, "paras"][i] for a, i in pkeys])
    np.savez(cache, CE=CE, PE=PE, pkeys=np.array(pkeys))
cidx = {a: i for i, a in enumerate(cand_ids)}
pidx = {}
for j, (a, i) in enumerate(pkeys):
    pidx.setdefault(a, []).append(j)
R["n_candidate_articles"], R["n_paragraphs"] = len(cand_ids), len(pkeys)

def part_vecs(a, paras, frac):
    """word-weighted mean of paragraph embeddings for seen / unseen parts (split as in step 1)"""
    w = np.array([len(p.split()) for p in paras], float); cw = np.cumsum(w) / w.sum()
    k = max(1, min(len(paras) - 1, int(np.searchsorted(cw, frac, side="right"))))
    E = PE[pidx[a]]
    v = [(E[:k] * w[:k, None]).sum(0), (E[k:] * w[k:, None]).sum(0)]
    return np.stack([x / np.linalg.norm(x) for x in v]), E[:k], E[k:]

BINS = [(5, 25), (25, 50), (50, 75), (75, 99.99)]
rows = []
for r in f.itertuples():
    cands = [c for c in r.article_ids_inview if c in cidx and c != r.prevA]
    if r.clk not in cands or len(cands) < 3 or r.prevS <= 5:
        continue
    full = r.prevS >= 100
    splits = [((lo + hi) / 2 / 100, (lo, hi)) for lo, hi in BINS] if full else \
             [(r.prevS / 100, next(b_ for b_ in BINS if b_[0] < r.prevS <= b_[1]))]
    C = CE[[cidx[c] for c in cands]]; ci = cands.index(r.clk)
    for frac, b_ in splits:
        V, Es, Eu = part_vecs(r.prevA, r.paras, frac)
        S = C @ V.T
        diff = S[:, 1] - S[:, 0]                                  # mean-pooled parts (primary)
        dmax = (C @ Eu.T).max(1) - (C @ Es.T).max(1)              # best-matching paragraph (secondary)
        oth = lambda d: d[ci] - np.delete(d, ci).mean()
        rows.append(dict(user=r.user_id, scroll=r.prevS, rt=r.prevR, full=full, bin=f"{b_[0]}-{int(b_[1])}",
                         delta=oth(diff), delta_max=oth(dmax), rank=(diff[ci] > np.delete(diff, ci)).mean(),
                         sim_whole=oth(C @ (V.sum(0) / np.linalg.norm(V.sum(0))))))
P = pd.DataFrame(rows)
P.to_csv(f"{OUT}/followup_pairs.csv", index=False)
R["n_pairs_partial"], R["n_pairs_placebo"] = int((~P.full).sum()), int(P.full.sum())

# sanity check: clicked item should be closer to the previous article as a whole than other candidates are
x = P[~P.full].sim_whole
R["sanity_clicked_minus_others_sim_to_prev_article"] = dict(mean=float(x.mean()), se=float(x.sem()), frac_pos=float((x > 0).mean()))

P["bin"] = pd.Categorical(P.bin, [f"{lo}-{int(hi)}" for lo, hi in BINS], ordered=True)
for col in ("delta", "delta_max"):
    T = P.groupby(["bin", "full"], observed=True).agg(mean=(col, "mean"), sem=(col, "sem"), n=(col, "size")).reset_index()
    R[f"{col}_by_bin"] = T.round(6).astype(str).to_dict("records")
    dd = {}
    for bn, x in P.groupby("bin", observed=True):
        a, c = x[~x.full][col], x[x.full][col]
        dd[bn] = dict(partial=float(a.mean()), placebo=float(c.mean()), diff=float(a.mean() - c.mean()),
                      se=float(np.sqrt(a.var() / len(a) + c.var() / len(c))), mannwhitney_p=float(stats.mannwhitneyu(a, c).pvalue))
    R[f"did_{col}"] = dd
    # pooled over bins: partial readers minus placebo
    a = P[~P.full].groupby("bin", observed=True)[col].mean(); c = P[P.full].groupby("bin", observed=True)[col].mean()
    R[f"pooled_{col}_partial_minus_placebo_unweighted_bins"] = float((a - c).mean())

    if col == "delta":
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for k, (fl, lab) in enumerate([(False, "stopped here (partial read)"), (True, "read to 100% (placebo, same split)")]):
            t = T[T.full == fl]
            ax.errorbar(np.arange(len(t)) + (k - .5) * .1, t["mean"], yerr=1.96 * t["sem"], fmt="o-", capsize=4, label=lab)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(BINS)), [f"{lo}-{int(hi)}" for lo, hi in BINS])
        ax.set_xlabel("split point in previous article (scroll %)"); ax.set_ylabel("clicked − other candidates")
        ax.set_title("e5-base: sim. to unseen part minus sim. to seen part"); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{OUT}/fig_unseen_pull_e5.png", dpi=130)

json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
