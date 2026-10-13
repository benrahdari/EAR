"""Step 3: offline re-ranking eval, whole-item vs partial exposure x category vs semantic units.

Each impression in the split is re-ranked by a profile built from the user's history (the 3 weeks before),
using one relevance model for every arm: score = <profile, candidate>. Metrics per impression: AUC, MRR, nDCG@5.

usage: python eval.py DATA_DIR SPLIT OUT_DIR [EMB.npz]     (no EMB -> category arms only)
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

D, SPLIT, OUT = sys.argv[1:4]
EMB = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(OUT, exist_ok=True)
WPM = 230
rng = np.random.default_rng(0)

A = pd.read_parquet(f"{D}/articles.parquet", columns=["article_id", "body", "category"]).set_index("article_id")
A["paras"] = A.body.fillna("").str.split("\n").apply(lambda p: [x for x in p if x.strip()])
A["n_words"] = A.paras.apply(lambda p: sum(len(x.split()) for x in p))
cat_code = {c: i for i, c in enumerate(sorted(A.category.unique()))}
A["cat"] = A.category.map(cat_code)
NC = len(cat_code)

h = pd.read_parquet(f"{D}/{SPLIT}/history.parquet").set_index("user_id")
b = pd.read_parquet(f"{D}/{SPLIT}/behaviors.parquet",
                    columns=["impression_id", "user_id", "article_ids_inview", "article_ids_clicked"])

# ---------- exposure estimate per history read: fraction of the text seen, top-down ----------
def exposure(scroll, rt, nw, variant):
    f_sc = np.where(np.isnan(scroll), np.nan, scroll / 100)
    f_rt = np.where(np.isnan(rt) | (nw == 0), np.nan, rt / 60 * WPM / np.maximum(nw, 1))
    # scroll+readtime: what was on screen AND plausibly read in the time spent; readtime alone is a dwell-time control
    f = {"scroll": f_sc, "readtime": f_rt}.get(variant, np.fmin(f_sc, f_rt))
    return np.clip(np.where(np.isnan(f), 1.0, f), 0, 1)  # no signal -> treat as fully read

# ---------- semantic: per-article prefix sums of word-weighted paragraph embeddings ----------
SEM = EMB is not None
if SEM:
    z = np.load(EMB)
    CE = z["CE"].astype(np.float32); cidx = {a: i for i, a in enumerate(z["cand"])}
    PE = z["PE"].astype(np.float32); pkeys = z["pkeys"]
    if os.environ.get("CENTER", "1") == "1":
        # e5 embeddings share a large common direction (anisotropy); averaging many reads converges to it and
        # swamps user-specific signal, so remove the corpus mean from both sides (Mu & Viswanath 2018 style)
        PE -= PE.mean(0); CE -= CE.mean(0)
    starts = np.r_[0, np.flatnonzero(np.diff(pkeys[:, 0])) + 1]
    art_rows = {int(pkeys[s, 0]): (s, e) for s, e in zip(starts, np.r_[starts[1:], len(pkeys)])}
    # Cum[a][k] = sum_{i<k} w_i e_i, W[a][k] = sum_{i<k} w_i, pos[a] = start position of each paragraph
    pre = {}
    for a, (s, e) in art_rows.items():
        w = np.array([len(x.split()) for x in A.at[a, "paras"]], float)
        if w.sum() == 0:
            continue
        w /= w.sum()
        cum = np.vstack([np.zeros(PE.shape[1], np.float32), np.cumsum(PE[s:e] * w[:, None].astype(np.float32), 0)])
        pre[a] = (cum, np.r_[0, np.cumsum(w)], np.cumsum(w) - w)

SEM_ARMS = ["S_whole", "S_whole_engagement_wt", "S_partial_content", "S_partial", "S_unseen"]
CAT_ARMS = ["C_whole", "C_partial"]
VARIANTS = ["scroll", "scroll+readtime", "readtime"]

def profiles(user):
    """returns {arm_variant: vector} for one user"""
    r = h.loc[user]
    aids = np.asarray(r.article_id_fixed, dtype=np.int64)
    sc = np.asarray(r.scroll_percentage_fixed, dtype=float); rt = np.asarray(r.read_time_fixed, dtype=float)
    keep = np.isin(aids, A.index.values)
    aids, sc, rt = aids[keep], sc[keep], rt[keep]
    nw = A.n_words.reindex(aids).values.astype(float)
    cats = A.cat.reindex(aids).values
    out = {"C_whole": np.bincount(cats, minlength=NC).astype(float)}
    for v in VARIANTS:
        f = exposure(sc, rt, nw, v)
        out[f"C_partial|{v}"] = np.bincount(cats, weights=f, minlength=NC)
        if SEM:
            acc = {k: np.zeros(PE.shape[1], np.float32) for k in SEM_ARMS}
            for a, fi in zip(aids, f):
                if a not in pre:
                    continue
                cum, W, pos = pre[a]
                k = max(1, int((pos < fi - 1e-9).sum())) if fi < 1 else len(pos)
                whole, seen = cum[-1], cum[k]
                acc["S_whole"] = acc["S_whole"] + whole
                acc["S_whole_engagement_wt"] = acc["S_whole_engagement_wt"] + fi * whole
                acc["S_partial_content"] = acc["S_partial_content"] + seen / W[k]
                acc["S_partial"] = acc["S_partial"] + seen
                if k < len(pos):
                    acc["S_unseen"] = acc["S_unseen"] + (whole - seen) / (1 - W[k])
            for k_, val in acc.items():
                out[f"{k_}|{v}"] = val
    out["S_whole"] = out.pop("S_whole|scroll", None) if SEM else None
    out.pop("S_whole|scroll+readtime", None); out.pop("S_whole|readtime", None)
    return {k: v for k, v in out.items() if v is not None}, float(np.mean(exposure(sc, rt, nw, "scroll"))), len(aids)

def metrics(s, y):
    s = s + rng.uniform(0, 1e-9, len(s))  # random tie-break (category scores tie a lot)
    pos, neg = y == 1, y == 0
    r = rankdata(s)
    auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    order = np.argsort(-s); yo = y[order]
    mrr = (yo / np.arange(1, len(y) + 1)).sum() / pos.sum()
    dcg = (yo[:5] / np.log2(np.arange(2, min(5, len(y)) + 2))).sum()
    idcg = (np.sort(y)[::-1][:5] / np.log2(np.arange(2, min(5, len(y)) + 2))).sum()
    return auc, mrr, dcg / idcg

rows = []
users = b.user_id.unique()
LIMIT = int(os.environ.get("USER_LIMIT", "0"))
for n_u, (u, g) in enumerate(b.groupby("user_id")):
    if LIMIT and n_u >= LIMIT:
        break
    if u not in h.index:
        continue
    P, mean_f, n_hist = profiles(u)
    for imp in g.itertuples():
        cands = [c for c in imp.article_ids_inview if c in A.index and (not SEM or c in cidx)]
        clicked = set(imp.article_ids_clicked)
        y = np.array([c in clicked for c in cands], int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        cc = A.cat.reindex(cands).values
        row = dict(user=u, imp=imp.impression_id, n_cand=len(cands), user_mean_scroll_exposure=mean_f, n_hist=n_hist)
        if SEM:
            C = CE[[cidx[c] for c in cands]]
        for arm, vec in P.items():
            s = vec[cc] / max(vec.sum(), 1e-9) if arm.startswith("C_") else C @ vec
            row.update({f"{arm}:{m}": x for m, x in zip(("auc", "mrr", "ndcg5"), metrics(np.asarray(s, float), y))})
        row["random:auc"], row["random:mrr"], row["random:ndcg5"] = metrics(np.zeros(len(y)), y)
        rows.append(row)
    if n_u % 2000 == 0:
        print(n_u, len(users), len(rows), flush=True)

R = pd.DataFrame(rows)
R.to_parquet(f"{OUT}/per_impression_{SPLIT}.parquet")

# ---------- summary with user-clustered bootstrap ----------
arms = sorted({c.split(":")[0] for c in R.columns if ":" in c})
U = R.groupby("user")
def boot_diff(col_a, col_b, B=1000):
    d = (R[col_a] - R[col_b]).groupby(R.user).agg(["sum", "count"]).values
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx, 0].sum(1) / d[idx, 1].sum(1)
    return float(d[:, 0].sum() / d[:, 1].sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

S = {"split": SPLIT, "n_impressions": len(R), "n_users": int(R.user.nunique()),
     "mean": {a: {m: float(R[f"{a}:{m}"].mean()) for m in ("auc", "mrr", "ndcg5")} for a in arms}}
pairs = [(f"C_partial|{v}", "C_whole") for v in VARIANTS]
if SEM:
    for v in VARIANTS:
        pairs += [(f"S_partial|{v}", "S_whole"), (f"S_partial_content|{v}", "S_whole"),
                  (f"S_whole_engagement_wt|{v}", "S_whole"), (f"S_unseen|{v}", "S_whole")]
    pairs += [("S_whole", "C_whole"), ("S_partial|scroll", "C_partial|scroll")]
S["paired_diff_auc"] = {f"{a} - {c}": dict(zip(("diff", "ci_lo", "ci_hi"), boot_diff(f"{a}:auc", f"{c}:auc"))) for a, c in pairs}
# where partial exposure should matter most: users who usually stop early
R["exposure_tercile"] = pd.qcut(R.user_mean_scroll_exposure.rank(method="first"), 3, labels=["low", "mid", "high"])
S["by_user_exposure_tercile_auc"] = {str(t): {a: float(x[f"{a}:auc"].mean()) for a in arms} | {"n": len(x)}
                                     for t, x in R.groupby("exposure_tercile", observed=True)}
json.dump(S, open(f"{OUT}/summary_{SPLIT}{os.environ.get('TAG', '')}{'' if SEM else '_category_only'}.json", "w"), indent=1)
print(json.dumps(S, indent=1))
