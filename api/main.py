from fastapi import FastAPI
from pydantic import BaseModel

from src.search import HSSearch
from src.rerank import rerank as rerank_llm   
from src.expand import expand_query
from src.db import init_db, log_pencarian      

app = FastAPI(title="HS Code Search API")

POOL_PER_CABANG = 30

# bikin mesin pencari SEKALI pas api nyala (muat model + data + embedding).
# ditaro di luar fungsi biar gak muat ulang tiap user nyari
mesin = HSSearch()

# siapin tabel database sekali pas api nyala (kalau belum ada, dibikin)
init_db()
 

# formulir data yang dikirim streamlit: nama barang + mau berapa hasil
class Permintaan(BaseModel):
    description: str
    top_k: int = 5


@app.get("/")
def cek():
    # buat ngecek api hidup apa nggak.
    return {"status": "ok", "jumlah_kode": len(mesin.df)}


@app.post("/search")
def search(req: Permintaan):
    # rapihin query: huruf kecil semua + buang spasi pinggir
    query = req.description.lower().strip()

    # 1. cari pakai query asli
    kandidat_asli = mesin.cari_kode(query, top_k=POOL_PER_CABANG)

    # 2. bersihin query pakai llm, terus cari LAGI pakai query bersihnya
    q_expanded = expand_query(query)
    kandidat_exp = mesin.cari_kode(q_expanded, top_k=POOL_PER_CABANG)

    # 3. gabung dua hasil + buang double
    #    triknya: tiap kandidat ditaro pakai kode sbg kunci, jadi kalau ada
    #    kode yang sama, yang belakangan numpuk yang depan
    pool = {}
    for k in kandidat_asli + kandidat_exp:
        pool[k["hs_code"]] = k
    kandidat = list(pool.values())

    # 4. rerank pakai query ASLI (bukan expanded) -> llm yang nentuin urutan paling cocok
    kandidat = rerank_llm(query, kandidat)

    # 5. ambil top_k teratas (sesuai slider user), kirim ke streamlit
    hasil = kandidat[:req.top_k]

    # 6. catat ke database. dibungkus try/except biar kalau mencatat gagal,
    #    app tetep jalan (jangan sampai fitur logging matiin pencarian).
    try:
        log_pencarian(query, hasil)
    except Exception as e:
        print(f"[WARN] gagal catat riwayat: {str(e)[:70]}")

    return {"query": query, "hasil": hasil}