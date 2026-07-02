import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

# impor komponen APP LANGSUNG -
from src.search import HSSearch
from src.expand import expand_query
from src.rerank import rerank as rerank_app

load_dotenv()

# config
KS = [1, 3, 5]
NDIGIT = 6          # level jenis produk (6-digit)
POOL = 30           # sama kaya POOL_PER_CABANG di main.py
DELAY_ANTAR_QUERY = 2.0   # jeda detik antar query

BASE = os.path.dirname(os.path.abspath(__file__))     # folder experiments/
EVAL_SET = os.path.join(BASE, "eval_set_bilingual.csv")
CHECKPOINT = os.path.join(BASE, "eval_llm_app_checkpoint.json")
RESULTS = os.path.join(BASE, "eval_llm_app_results.csv")

# load eval set
ev = pd.read_csv(EVAL_SET, dtype=str)
truths = ev["hs_code"].tolist()
queries_id = ev["query_id"].tolist()

# mesin pencari (muat model + data + desc_emb)
mesin = HSSearch()


# full pipeline NIRU main.py 
def full_pipeline(query):
    q = query.lower().strip()                 # fix kapital
    kand_asli = mesin.cari_kode(q, top_k=POOL)      # cabang 1: query ASLI
    q_exp = expand_query(q)                          # bersihin query pakai llm
    kand_exp = mesin.cari_kode(q_exp, top_k=POOL)    # cabang 2: query EXPANDED
    # gabung + dedup pakai kode sbg kunci (sama kaya app)
    pool = {}
    for k in kand_asli + kand_exp:
        pool[k["hs_code"]] = k
    kandidat = list(pool.values())
    ranked = rerank_app(q, kandidat)          # rerank pakai query ASLI

    # deteksi rerank GAGAL (kena limit): kalau top 5 nggak ada alasan sama sekali,
    # lempar error biar query ini nggak disimpan, nanti diulang.
    ada_alasan = any(k.get("alasan") for k in ranked[:5])
    if not ada_alasan:
        raise RuntimeError("rerank kemungkinan kena limit (top 5 tanpa alasan)")

    return [k["hs_code"] for k in ranked]


def hit_rate(ranked_per_query, k):
    hit = 0
    for ranked, truth in zip(ranked_per_query, truths):
        if truth[:NDIGIT] in [c[:NDIGIT] for c in ranked[:k]]:
            hit += 1
    return hit / len(truths)

# baseline: embedding doang (query asli, tanpa llm)
print("== embedding doang ==")
emb_only = [[k["hs_code"] for k in mesin.cari_kode(q.lower().strip(), top_k=POOL)]
            for q in queries_id]

# full pipeline, pakai checkpoint biar bisa lanjut kalau kena limit
ckpt = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    print(f"   (lanjut dari checkpoint: {len(ckpt)}/{len(queries_id)} query udah kelar)")

print("== full pipeline (niru app: 2 cabang + rerank nomor-JSON) ==")
for i, q in enumerate(queries_id):
    key = str(i)
    if key in ckpt:
        continue
    try:
        hasil_q = full_pipeline(q)
    except Exception as e:
        print(f"\n[STOP] query ke-{i+1} gagal: {str(e)[:60]}")
        print("       benerin limit dulu, terus jalanin script ini lagi buat lanjut.")
        break
    ckpt[key] = hasil_q
    with open(CHECKPOINT, "w") as f:  
        json.dump(ckpt, f)
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{len(queries_id)} query selesai")
    time.sleep(DELAY_ANTAR_QUERY)

# angka cuma dihitung kalau semua query lengkap, biar nggak misleading
if len(ckpt) < len(queries_id):
    raise SystemExit(f"\nbaru {len(ckpt)}/{len(queries_id)} query kelar. "
                     f"jalanin lagi nanti buat nerusin.")

full = [ckpt[str(i)] for i in range(len(queries_id))]


# tabel hasil
hasil = {
    "embedding doang": emb_only,
    "+ expansion + rerank (niru app)": full,
}
rows = []
for name, ranked in hasil.items():
    r = {"kondisi": name}
    for k in KS:
        r[f"hit@{k}"] = round(hit_rate(ranked, k), 4)
    rows.append(r)
res = pd.DataFrame(rows)

print(f"\n{'='*70}\nHASIL — query INDONESIA, level {NDIGIT}-digit (NIRU APP)\n{'='*70}")
print(res.to_string(index=False))
res.to_csv(RESULTS, index=False)
print(f"\ndisimpan ke: {RESULTS}")

base = hit_rate(emb_only, 5)
app = hit_rate(full, 5)
print(f"\nhit@5: embedding {base:.0%}  ->  app (niru) {app:.0%}")
print(f"lift : {(app-base)*100:+.1f} poin")