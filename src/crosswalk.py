import re
import pandas as pd
from src.config import PROCESSED_DATA_DIR

_DATA = PROCESSED_DATA_DIR / "hscode_allversions.csv"
_df = None
ERA_LAMA = ["2017-2021", "2012-2016", "2009-2011", "1999-2008"]   # terbaru -> terlama


def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(_DATA, dtype={"hs_code": str})
    return _df


def _kata_isi(teks):
    # ambil kata isi: huruf saja, panjang >= 4. ini otomatis buang kata sambung
    # pendek (and, or, of, in, the, oth) tapi tetep simpen kata penting yang
    # membedakan barang (fresh, frozen, live, dried, dll).
    return {w for w in re.findall(r"[a-z]+", str(teks).lower()) if len(w) >= 4}


def _skor_mirip(desc_a, desc_b):
    # seberapa mirip dua deskripsi = jumlah kata isi yang sama / gabungan (jaccard).
    # 0 = nggak ada kata sama; makin tinggi makin mirip. murni hitung kata, no model.
    a, b = _kata_isi(desc_a), _kata_isi(desc_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_crosswalk(kode_2022, deskripsi_2022="", max_per_era=3):
    """dict {era: [ {hs_code, description, match} ]} buat tiap era lama.

    deskripsi_2022 = deskripsi kode yang dipilih user; dipakai buat ngurutin
    kandidat pas fallback 4-digit biar yang paling mirip barangnya naik ke atas.
    """
    df = _load()
    hs6, hs4 = kode_2022[:6], kode_2022[:4]
    hasil = {}
    for era in ERA_LAMA:
        sub = df[df["version"] == era]
        cocok = sub[sub["hs_code"].str[:6] == hs6]
        match_type = "tepat (6-digit)"
        if cocok.empty:
            # 6-digit nggak ada (kode berubah pas revisi) -> mundur ke heading 4-digit.
            cocok = sub[sub["hs_code"].str[:4] == hs4].copy()
            match_type = "perkiraan (heading 4-digit)"
            # urutin kandidat heading berdasar kemiripan deskripsi ke barang asli.
            # stable sort: kalau deskripsi_2022 kosong / semua skor 0, urutan tetap
            # apa adanya (sama kayak ambil baris pertama), jadi aman.
            if deskripsi_2022 and not cocok.empty:
                cocok["_skor"] = cocok["description"].apply(
                    lambda d: _skor_mirip(deskripsi_2022, d))
                cocok = cocok.sort_values("_skor", ascending=False, kind="stable")
        hasil[era] = [{"hs_code": r["hs_code"], "description": r["description"], "match": match_type}
                      for _, r in cocok.head(max_per_era).iterrows()]
    return hasil 