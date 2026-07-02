import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)   # bikin folder models kalau belum ada 

df = pd.read_csv(PROCESSED_DATA_DIR / "hscode_clean.csv", dtype={"hs_code": str})
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")   

print("meng-embed", len(df), "deskripsi (sekali aja)...")
emb = model.encode(df["description"].tolist(), show_progress_bar=True)  # ubah semua deskripsi jadi angka

out = MODELS_DIR / "desc_emb.npy"
np.save(out, emb)   # simpan angkanya ke file, biar gak ngitung ulang tiap app jalan
print("selesai. tersimpan:", out, emb.shape)