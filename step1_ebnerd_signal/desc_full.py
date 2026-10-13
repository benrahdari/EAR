"""Step 1 descriptive analysis on full EB-NeRD, memory-efficient (pyarrow flatten instead of pandas explode)."""
import sys, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

D, OUT = sys.argv[1], sys.argv[2]
arts = pd.read_parquet(f"{D}/articles.parquet", columns=["article_id", "body", "image_ids"])
arts["n_words"] = arts.body.fillna("").str.split().str.len()
arts["n_imgs"] = arts.image_ids.apply(lambda x: 0 if x is None else len(x))
amax = int(arts.article_id.max()) + 1
nw = np.zeros(amax, np.int32); ni = np.full(amax, -1, np.int32)
nw[arts.article_id] = arts.n_words; ni[arts.article_id] = arts.n_imgs
del arts

# stream row groups: exact counts/shares over all reads, plus a 2% random sample for medians and correlations
rng = np.random.default_rng(0)
seen_users = set(); N = 0; n_ok = 0; n_nan = 0; n100 = 0; n50 = 0
usum = {}; samp = []
for split in ("train", "validation"):
    f = pq.ParquetFile(f"{D}/{split}/history.parquet")
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["user_id", "scroll_percentage_fixed", "article_id_fixed", "read_time_fixed"])
        u = t["user_id"].to_numpy()
        keep = np.array([x not in seen_users for x in u.tolist()])
        t = t.filter(keep); u = u[keep]; seen_users.update(u.tolist())
        lens = pc.list_value_length(t["article_id_fixed"]).to_numpy()
        user = np.repeat(u, lens)
        aid = pc.list_flatten(t["article_id_fixed"]).to_numpy()
        sc = pc.list_flatten(t["scroll_percentage_fixed"]).to_numpy(zero_copy_only=False).astype(np.float32)
        rt = pc.list_flatten(t["read_time_fixed"]).to_numpy(zero_copy_only=False).astype(np.float32)
        del t
        N += len(aid); ok = nw[aid] > 0
        n_nan += int(np.isnan(sc[ok]).sum())
        m = ok & ~np.isnan(sc); n_ok += int(m.sum())
        user, aid, sc, rt = user[m], aid[m], sc[m], rt[m]
        n100 += int((sc == 100).sum()); n50 += int((sc < 50).sum())
        # per-user moments for the user share of variance
        uu, inv = np.unique(user, return_inverse=True)
        c = np.bincount(inv); s1 = np.bincount(inv, sc); s2 = np.bincount(inv, sc.astype(np.float64) ** 2)
        for k, x in enumerate(uu.tolist()):
            usum[x] = (c[k], s1[k], s2[k])
        pick = rng.random(len(sc)) < 0.02
        samp.append(pd.DataFrame(dict(sc=sc[pick], rt=rt[pick], w=nw[aid[pick]], ni=ni[aid[pick]], user=user[pick])))
        print(split, g, N, flush=True)
R = {"users": len(seen_users), "history_reads": N, "reads_with_scroll": n_ok,
     "scroll_missing": n_nan / (n_ok + n_nan), "scroll_eq_100": n100 / n_ok, "scroll_lt_50": n50 / n_ok}
U = np.array(list(usum.values())); U = U[U[:, 0] >= 20]
umean = U[:, 1] / U[:, 0]; tot_mean = U[:, 1].sum() / U[:, 0].sum(); tot_var = U[:, 2].sum() / U[:, 0].sum() - tot_mean ** 2
R["user_share_of_scroll_variance"] = float(umean.var() / tot_var)
df = pd.concat(samp, ignore_index=True); del samp, usum
R["sample_reads_for_medians"] = len(df)
sc, rt, w = df.sc.values, df.rt.values, df.w.values
frac_rt = np.clip(rt / 60 * 230 / w, 0, 1)
rng = np.random.default_rng(0); sub = np.arange(len(sc))
R["spearman_scroll_readtime"] = float(stats.spearmanr(sc[sub], rt[sub]).statistic)
R["spearman_scroll_impliedfrac"] = float(stats.spearmanr(sc[sub], frac_rt[sub]).statistic)
df["fr"] = frac_rt
df["len_bin"] = pd.qcut(df.w, 5)
bl = df.groupby("len_bin", observed=True)
R["by_length"] = pd.DataFrame(dict(median_scroll=bl.sc.median(), share_100=bl.sc.apply(lambda x: (x == 100).mean()),
                                   median_rt=bl.rt.median(), median_frac_rt=bl.fr.median(), n=bl.size())).round(3).reset_index().astype(str).to_dict("records")
df["img_bin"] = pd.cut(df.ni, [-1, 0, 1, 3, 1000], labels=["0", "1", "2-3", "4+"])
bi = df.groupby("img_bin", observed=True)
R["by_images"] = pd.DataFrame(dict(median_scroll=bi.sc.median(), share_100=bi.sc.apply(lambda x: (x == 100).mean()), n=bi.size())).round(3).reset_index().astype(str).to_dict("records")
L = df[df.w >= 300]
R["long_articles_scroll100_under10s"] = float(((L.sc == 100) & (L.rt < 10)).mean())
R["long_articles_scroll_lt50_over60s"] = float(((L.sc < 50) & (L.rt > 60)).mean())
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(sc, bins=50, color="#4C72B0"); ax[0].set_title("Scroll % per article read (full EB-NeRD)"); ax[0].set_xlabel("scroll %")
ax[1].plot(range(5), bl.sc.median(), "o-", label="median scroll %")
ax[1].plot(range(5), bl.sc.apply(lambda x: (x == 100).mean() * 100), "s-", label="% reads at 100")
ax[1].set_xticks(range(5), [f"{int(i.left)}-{int(i.right)}" for i in bl.groups.keys()], rotation=20)
ax[1].set_xlabel("article length (words, quintiles)"); ax[1].legend(); ax[1].set_title("Scroll vs. length")
ax[2].hexbin(sc[sub] / 100, frac_rt[sub], gridsize=30, bins="log", cmap="Blues")
ax[2].set_xlabel("scroll fraction"); ax[2].set_ylabel("read-time-implied fraction (230 wpm)"); ax[2].set_title("Two estimates of fraction read")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_scroll_profile_full.png", dpi=130)
json.dump(R, open(f"{OUT}/results_full.json", "w"), indent=1)
print(json.dumps(R, indent=1))
