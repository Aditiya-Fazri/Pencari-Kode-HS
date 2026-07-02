import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.config import MODELS_DIR   # buat ambil desc_emb.npy punya app

# config
DATA_CSV = "data/processed/hscode_clean.csv"
EVAL_CSV = "experiments/eval_set_bilingual.csv"
CACHE_DIR = "cache_emb"

APP_MODEL = "paraphrase-multilingual-mpnet-base-v2"   # model yang dipakai app
EMB_MODELS = {
    "paraphrase-multilingual-mpnet-base-v2": APP_MODEL,
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2":                    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2":                     "all-mpnet-base-v2",
}
KS = [1, 3, 5, 10]
DIGITS = [6, 8]

os.makedirs(CACHE_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV, dtype={"hs_code": str})
codes = df["hs_code"].tolist()
descs = df["description"].fillna("").tolist()

ev = pd.read_csv(EVAL_CSV, dtype=str)
truths = ev["hs_code"].tolist()
# LOWERCASE + STRIP query, biar SAMA PERSIS kaya app (main.py: q.lower().strip())
queries = {
    "EN": [str(q).lower().strip() for q in ev["query_en"].tolist()],
    "ID": [str(q).lower().strip() for q in ev["query_id"].tolist()],
}
print(f"kode di index : {len(codes)}")
print(f"query eval    : {len(truths)} (x2 bahasa)\n")


def add_prefix(hf, texts, kind):
    if "intfloat" in hf and "e5" in hf:
        tag = "query: " if kind == "query" else "passage: "
        return [tag + t for t in texts]
    return texts


def get_desc_emb(disp, hf, model):
    # khusus model app (paraphrase-multilingual-mpnet-base-v2): pakai desc_emb.npy punya app
    if hf == APP_MODEL:
        app_npy = MODELS_DIR / "desc_emb.npy"
        if app_npy.exists():
            print(f"  pakai desc_emb.npy punya app buat {disp} (dijamin sama kaya app)")
            return np.load(app_npy)
    # model lain: embed sendiri, cache
    path = os.path.join(CACHE_DIR, f"{hf.replace('/', '_')}.npy")
    if os.path.exists(path):
        return np.load(path)
    print(f"  embed {len(descs)} deskripsi pakai {disp} (sekali aja)...")
    emb = model.encode(add_prefix(hf, descs, "passage"), show_progress_bar=True, batch_size=64)
    np.save(path, emb)
    return emb


def ranked_codes(q_emb, desc_emb, top_k=max(KS)):
    skor = cosine_similarity(q_emb, desc_emb)
    return [[codes[i] for i in row.argsort()[::-1][:top_k]] for row in skor]


def hit_rate(ranked_per_query, k, nd):
    hit = 0
    for ranked, truth in zip(ranked_per_query, truths):
        if truth[:nd] in [c[:nd] for c in ranked[:k]]:
            hit += 1
    return hit / len(truths)


hasil = {}
for disp, hf in EMB_MODELS.items():
    print(f"== {disp} ==")
    try:
        model = SentenceTransformer(hf, trust_remote_code=True)
        desc_emb = get_desc_emb(disp, hf, model)
        hasil[disp] = {}
        for lang, qs in queries.items():
            q_emb = model.encode(add_prefix(hf, qs, "query"), batch_size=64)
            hasil[disp][lang] = ranked_codes(q_emb, desc_emb)
    except Exception as e:
        print(f"   (GAGAL, di-skip: {str(e)[:90]})")


# tabel hasil   
rows = []
for nd in DIGITS:
    for lang in ["ID", "EN"]:
        for disp in hasil:
            r = {"level": f"{nd}-digit", "bahasa": lang, "model": disp}
            for k in KS:
                r[f"hit@{k}"] = round(hit_rate(hasil[disp][lang], k, nd), 4)
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv("eval_emb_results.csv", index=False)

for lang in ["ID", "EN"]:
    print(f"\n{'='*60}\nquery {lang} — level 6-digit\n{'='*60}")
    sub = res[(res["level"] == "6-digit") & (res["bahasa"] == lang)]
    print(sub.drop(columns=["level", "bahasa"]).to_string(index=False))

print("\nlengkap disimpan ke: eval_emb_results.csv")