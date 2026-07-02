import os
import sqlite3
from datetime import datetime

# simpen database di root project (satu folder di atas src/), nama: riwayat.db
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "riwayat.db")


def init_db():
    # bikin tabel sekali di awal (kalau belum ada). dipanggil pas api nyala.
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pencarian (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT,
            query TEXT,
            kode_teratas TEXT,
            deskripsi_teratas TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_pencarian(query, hasil):
    # catat 1 baris tiap pencarian. ambil hasil peringkat teratas (top 1) buat disimpan.
    kode = hasil[0]["hs_code"] if hasil else ""
    desk = hasil[0]["description"] if hasil else ""
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO pencarian (waktu, query, kode_teratas, deskripsi_teratas) VALUES (?, ?, ?, ?)",
        (waktu, query, kode, desk),
    )
    conn.commit()
    conn.close()


def ambil_riwayat(limit=50):
    # ambil pencarian terakhir (buat ditampilin di tab riwayat streamlit nanti)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT waktu, query, kode_teratas, deskripsi_teratas "
        "FROM pencarian ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
