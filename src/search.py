import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PROCESSED_DATA_DIR, MODELS_DIR


class HSSearch:
    def __init__(self,
                 csv_path=PROCESSED_DATA_DIR / "hscode_clean.csv",
                 emb_path=MODELS_DIR / "desc_emb.npy",
                 model_name="paraphrase-multilingual-mpnet-base-v2"):
        # semua dimuat sekali di sini
        self.df = pd.read_csv(csv_path, dtype={"hs_code": str})   # teks: kode + deskripsi
        self.desc_emb = np.load(emb_path)                         # angka: hasil embed dari build_index
        self.model = SentenceTransformer(model_name)              # memuat model embedding 

    def cari_kode(self, query, top_k=5):
        q_emb = self.model.encode([query])                 # ubah query menjadi embedding
        skor = cosine_similarity(q_emb, self.desc_emb)[0]  # bandingin ke semua deskripsi -> skor mirip
        # argsort ngasih nomor baris diurut skor kecil->besar, [::-1] dibalik biar besar dulu, terus ambil sebanyak top_k teratas. 
        idx_teratas = skor.argsort()[::-1][:top_k]
        hasil = []
        for i in idx_teratas:
            # balik ke csv buat ambil teks kode + deskripsi di baris yang skornya tinggi
            hasil.append({
                "hs_code": self.df.iloc[i]["hs_code"],
                "description": self.df.iloc[i]["description"],
                "score": round(float(skor[i]), 3),
            })
        return hasil