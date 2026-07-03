import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src.search import HSSearch
from src.expand import expand_query
from src.rerank import rerank
from src.crosswalk import get_crosswalk
from src.db import init_db, log_pencarian, ambil_riwayat

# berapa kandidat ditarik per cabang sebelum rerank
POOL_PER_CABANG = 30

st.set_page_config(page_title="Pencari Kode HS")


# muat mesin pencari sekali (model + data + embedding). cache_resource bikin ini gak diulang tiap user klik.
@st.cache_resource
def load_mesin():
    init_db()          # siapin tabel database sekali
    return HSSearch()

mesin = load_mesin()


# pipeline pencarian, sama persis kaya di api/main.py cuma dipanggil langsung
# (tanpa lewat api), biar streamlit bisa jalan sendiri pas di deploy.
def cari(query, top_k, api_key):
    kandidat_asli = mesin.cari_kode(query, top_k=POOL_PER_CABANG)   # cabang 1: query asli
    q_exp = expand_query(query, api_key=api_key)                    # bersihin query pakai llm
    kandidat_exp = mesin.cari_kode(q_exp, top_k=POOL_PER_CABANG)    # cabang 2: query expanded
    # gabung + buang double (kode sbg kunci)
    pool = {}
    for k in kandidat_asli + kandidat_exp:
        pool[k["hs_code"]] = k
    kandidat = rerank(query, list(pool.values()), api_key=api_key)  # rerank pakai query asli
    return kandidat[:top_k]


st.title("Pencari Kode HS")
st.caption("Isi nama barang, kalau ada tambahin deskripsinya biar hasilnya lebih tepat.")

# sidebar: API key Groq (diinput user, gak ditaro di code)
with st.sidebar:
    st.subheader("Groq API Key")
    api_key = st.text_input("Masukkan API key", type="password",
                            placeholder="gsk_...")

# dua tab: satu buat nyari, satu buat liat riwayat pencarian yang tersimpan di database
tab_cari, tab_riwayat = st.tabs(["Cari", "Riwayat"])

# TAB CARI
with tab_cari:
    nama = st.text_input("Nama barang")
    deskripsi = st.text_area("Deskripsi tambahan (opsional)",
                             placeholder="contoh: terbuat dari stainless steel")
    top_k = st.slider("Jumlah hasil", 1, 10, 5)

    if st.button("Cari"):
        # gabung nama + deskripsi jadi satu query (buang yang kosong)
        query = " ".join([b for b in [nama.strip(), deskripsi.strip()] if b])
        if not api_key:
            st.warning("Masukin Groq API key dulu di sidebar sebelah kiri.")
        elif query == "":
            st.warning("Isi dulu minimal nama barangnya.")
        else:
            with st.spinner("Pakai AI"):
                hasil = cari(query, top_k, api_key)
            st.session_state["hasil"] = hasil
            st.session_state["query"] = query
            # catat ke database (dibungkus try biar kalau gagal, app tetep jalan)
            try:
                log_pencarian(query, hasil)
            except Exception as e:
                print(f"[WARN] gagal nyatet riwayat: {str(e)[:70]}")

    # tampilin hasil (kalau udah pernah nyari)
    if "hasil" in st.session_state:
        hasil = st.session_state["hasil"]

        st.subheader("Hasil")
        df = pd.DataFrame(hasil)
        df["peringkat"] = range(1, len(df) + 1)
        df = df.rename(columns={"hs_code": "Kode HS", "description": "Deskripsi",
                                "alasan": "Keterangan"})
        kolom = ["peringkat", "Kode HS", "Deskripsi"]
        if "Keterangan" in df.columns:
            kolom.append("Keterangan")
        df = df[kolom].set_index("peringkat")
        st.table(df)

        # crosswalk: padanan versi tahun
        st.subheader("HS Code versi tahun")
        st.caption("Pilih salah satu kode di atas untuk melihat HS Code di tahun sebelumnya.")

        opsi = {f'{h["hs_code"]} — {h["description"][:60]}': h for h in hasil}
        pilih = st.selectbox("Kode HS 2022", list(opsi.keys()))
        h_pilih = opsi[pilih]

        cw = get_crosswalk(h_pilih["hs_code"], h_pilih["description"])
        baris = []
        for era, rows in cw.items():
            for r in rows:
                singkat = "tepat" if r["match"].startswith("tepat") else "perkiraan"
                baris.append({"versi": era, "kode hs": r["hs_code"],
                              "deskripsi": r["description"], "match": singkat})

        if baris:
            st.table(pd.DataFrame(baris))
        else:
            st.info("Tidak ada HS Code ditemukan di versi sebelumnya.")

# TAB RIWAYAT
with tab_riwayat:
    st.subheader("Riwayat Pencarian")
    st.caption("Semua pencarian yang pernah dilakukan, tersimpan di database (riwayat.db).")

    rows = ambil_riwayat(limit=50)
    if rows:
        df_riwayat = pd.DataFrame(rows, columns=["waktu", "query", "kode hs", "deskripsi"])
        st.table(df_riwayat)
    else:
        st.info("Belum ada pencarian yang tercatat.")