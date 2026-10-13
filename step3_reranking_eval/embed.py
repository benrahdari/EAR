"""Embed (a) every paragraph of every article in the users' histories and (b) every candidate article,
with intfloat/multilingual-e5-base. usage: python embed.py DATA_DIR SPLIT OUT.npz"""
import sys, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
D, SPLIT, OUT = sys.argv[1:4]
A = pd.read_parquet(f"{D}/articles.parquet").set_index("article_id")
A["paras"] = A.body.fillna("").str.split("\n").apply(lambda p: [x for x in p if x.strip()])
h = pd.read_parquet(f"{D}/{SPLIT}/history.parquet")
b = pd.read_parquet(f"{D}/{SPLIT}/behaviors.parquet")
hist = sorted(set(np.concatenate(h.article_id_fixed.values)) & set(A.index))
cand = sorted(set(np.concatenate(b.article_ids_inview.values)) & set(A.index))
pkeys = np.array([(a, i) for a in hist for i in range(len(A.at[a, "paras"]))])
m = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
enc = lambda xs, L: (setattr(m, "max_seq_length", L), m.encode(["query: " + x for x in xs], batch_size=64,
                     normalize_embeddings=True, show_progress_bar=True).astype(np.float16))[1]
CE = enc((A.loc[cand].title.fillna("") + ". " + A.loc[cand].subtitle.fillna("") + " " + A.loc[cand].body.fillna("")).tolist(), 512)
np.savez(OUT + ".cand.npz", CE=CE, cand=np.array(cand))
PE = enc([A.at[a, "paras"][i] for a, i in pkeys], 256)
np.savez(OUT, CE=CE, cand=np.array(cand), PE=PE, pkeys=pkeys)
print("done", CE.shape, PE.shape)
