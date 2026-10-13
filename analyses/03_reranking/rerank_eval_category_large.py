"""Category arms of the re-ranking evaluation, vectorized so it runs on EB-NeRD large (12.6M impressions).
Same profiles, exposure estimates and metrics as rerank_eval.py. usage: python rerank_eval_category_large.py DATA_DIR SPLIT OUT_DIR"""
import sys, os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

D, SPLIT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
WPM = 230
rng = np.random.default_rng(0)

art = pd.read_parquet(f"{D}/articles.parquet", columns=["article_id", "body", "category"])
art["n_words"] = art.body.fillna("").str.split().str.len()
cat_code = {c: i for i, c in enumerate(sorted(art.category.unique()))}
NC = len(cat_code)
amax = int(art.article_id.max()) + 1
CAT = np.full(amax, -1, np.int64); CAT[art.article_id.values] = art.category.map(cat_code).values
NW = np.zeros(amax); NW[art.article_id.values] = art.n_words.values
del art

# ---------- user profiles (one history row group at a time to bound memory) ----------
hf = pq.ParquetFile(f"{D}/{SPLIT}/history.parquet")
NU = hf.metadata.num_rows
uid = np.empty(NU, np.int64)
PROF = ("C_whole", "C_partial|scroll", "C_partial|scroll+readtime", "C_partial|readtime")
prof = {k: np.zeros((NU, NC), np.float32) for k in PROF}
user_mean_f = np.zeros(NU)

def exposure(sc, rt, nw, variant):
    f_sc = sc / 100
    f_rt = np.where(nw == 0, np.nan, rt / 60 * WPM / np.maximum(nw, 1))
    f = {"scroll": f_sc, "readtime": f_rt}.get(variant, np.fmin(f_sc, f_rt))
    return np.clip(np.where(np.isnan(f), 1.0, f), 0, 1)

off = 0
for rg in range(hf.num_row_groups):
    h = hf.read_row_group(rg, columns=["user_id", "article_id_fixed", "scroll_percentage_fixed", "read_time_fixed"])
    n = h.num_rows
    uid[off:off + n] = h.column("user_id").to_numpy()
    lens = pc.list_value_length(h.column("article_id_fixed")).to_numpy()
    ui = np.repeat(np.arange(n, dtype=np.int64), lens)
    a = pc.list_flatten(h.column("article_id_fixed")).to_numpy().astype(np.int64)
    sc = pc.list_flatten(h.column("scroll_percentage_fixed")).to_numpy(zero_copy_only=False).astype(np.float32)
    rt = pc.list_flatten(h.column("read_time_fixed")).to_numpy(zero_copy_only=False).astype(np.float32)
    del h
    ok = (a < amax) & (CAT[np.minimum(a, amax - 1)] >= 0)
    ui, a, sc, rt = ui[ok], a[ok], sc[ok], rt[ok]
    c, nw = CAT[a], NW[a]
    key = ui * NC + c
    prof["C_whole"][off:off + n] = np.bincount(key, minlength=n * NC).reshape(n, NC)
    for v in ("scroll", "scroll+readtime", "readtime"):
        f = exposure(sc, rt, nw, v)
        prof[f"C_partial|{v}"][off:off + n] = np.bincount(key, weights=f, minlength=n * NC).reshape(n, NC)
        if v == "scroll":
            user_mean_f[off:off + n] = np.bincount(ui, weights=f, minlength=n) / np.maximum(np.bincount(ui, minlength=n), 1)
    off += n
    del ui, a, sc, rt, c, nw, key
    print("history", rg, hf.num_row_groups, flush=True)
uix = {u: i for i, u in enumerate(uid)}
for k in prof:
    prof[k] /= np.maximum(prof[k].sum(1, keepdims=True), 1e-9)
ARMS = list(prof) + ["random"]

# ---------- score impressions, one row group at a time ----------
pf = pq.ParquetFile(f"{D}/{SPLIT}/behaviors.parquet")
per_user = []  # per-impression metrics aggregated later
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["impression_id", "user_id", "article_ids_inview", "article_ids_clicked"])
    n = pc.list_value_length(t.column("article_ids_inview")).to_numpy()
    imp = np.repeat(np.arange(t.num_rows), n)
    cand = pc.list_flatten(t.column("article_ids_inview")).to_numpy().astype(np.int64)
    clk_n = pc.list_value_length(t.column("article_ids_clicked")).to_numpy()
    clk = pc.list_flatten(t.column("article_ids_clicked")).to_numpy().astype(np.int64)
    clk_key = set(zip(np.repeat(np.arange(t.num_rows), clk_n).tolist(), clk.tolist()))
    y = np.fromiter(((i, x) in clk_key for i, x in zip(imp.tolist(), cand.tolist())), bool, len(imp)).astype(np.int8)
    users = np.array([uix.get(u, -1) for u in t.column("user_id").to_numpy()])
    keep = (users[imp] >= 0) & (cand < amax) & (CAT[np.minimum(cand, amax - 1)] >= 0)
    imp, cand, y = imp[keep], cand[keep], y[keep]
    df = pd.DataFrame({"imp": imp, "y": y})
    g = df.groupby("imp").y.agg(["sum", "size"])
    valid = g.index[(g["sum"] > 0) & (g["sum"] < g["size"])]
    m = np.isin(imp, valid)
    imp, cand, y = imp[m], cand[m], y[m]
    cc, uu = CAT[cand], users[imp]
    npos = np.bincount(imp, weights=y, minlength=t.num_rows); ntot = np.bincount(imp, minlength=t.num_rows)
    res = {"user": users[valid.values], "mean_f": user_mean_f[users[valid.values]]}
    for arm in ARMS:
        s = (prof[arm][uu, cc] if arm != "random" else np.zeros(len(imp))) + rng.uniform(0, 1e-6, len(imp))
        d = pd.DataFrame({"imp": imp, "s": s})
        r_asc = d.groupby("imp").s.rank(method="first").values
        r_desc = ntot[imp] - r_asc + 1
        rs = np.bincount(imp, weights=y * r_asc, minlength=t.num_rows)
        P, N = npos, ntot - npos
        auc = (rs - P * (P + 1) / 2) / np.maximum(P * N, 1)
        mrr = np.bincount(imp, weights=y / r_desc, minlength=t.num_rows) / np.maximum(P, 1)
        dcg = np.bincount(imp, weights=y * (r_desc <= 5) / np.log2(r_desc + 1), minlength=t.num_rows)
        idcg = np.array([(1 / np.log2(np.arange(2, min(int(p), 5) + 2))).sum() for p in P])
        nd = dcg / np.maximum(idcg, 1e-9)
        for k, x in (("auc", auc), ("mrr", mrr), ("ndcg5", nd)):
            res[f"{arm}:{k}"] = x[valid.values]
    per_user.append(pd.DataFrame(res))
    print(rg, pf.num_row_groups, sum(len(x) for x in per_user), flush=True)

R = pd.concat(per_user, ignore_index=True)
R.to_parquet(f"{OUT}/per_impression_category_{SPLIT}.parquet")

def boot_diff(col_a, col_b, B=500):
    d = (R[col_a] - R[col_b]).groupby(R.user).agg(["sum", "count"]).values
    bs = []
    for _ in range(B):
        w = rng.poisson(1, len(d))  # Poisson bootstrap over users (fast for ~800k users)
        bs.append((d[:, 0] * w).sum() / (d[:, 1] * w).sum())
    return float(d[:, 0].sum() / d[:, 1].sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

S = {"split": SPLIT, "n_impressions": len(R), "n_users": int(R.user.nunique()),
     "mean": {a_: {k: float(R[f"{a_}:{k}"].mean()) for k in ("auc", "mrr", "ndcg5")} for a_ in ARMS},
     "paired_diff_auc": {f"{a_} - C_whole": dict(zip(("diff", "ci_lo", "ci_hi"), boot_diff(f"{a_}:auc", "C_whole:auc")))
                         for a_ in ("C_partial|scroll", "C_partial|scroll+readtime", "C_partial|readtime")}}
R["t"] = pd.qcut(R.mean_f.rank(method="first"), 3, labels=["low", "mid", "high"])
S["by_user_exposure_tercile_auc"] = {str(k): {a_: float(x[f"{a_}:auc"].mean()) for a_ in ARMS} | {"n": len(x)}
                                     for k, x in R.groupby("t", observed=True)}
json.dump(S, open(f"{OUT}/summary_category_{SPLIT}.json", "w"), indent=1)
print(json.dumps(S, indent=1))
