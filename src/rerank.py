import os
import re
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _client(api_key=None):
    # sama kaya di expand: pakai api_key dari argumen kalau ada, kalau kosong dari .env
    return Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))


def _ambil_json(teks):
    # llm kadang ngasih teks tambahan / pakai ```json. tarik array jsonnya aja.
    m = re.search(r"\[.*\]", teks, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# kandidat = list dict {hs_code, description, score} dari HSSearch
# balikin list dict yang sama tapi udah diurutin ulang sama llm, + alasan.
def rerank(query, kandidat, api_key=None, retries=3):
    # tampilin kandidat pakai NOMOR (bukan kode). llm nanti jawab pakai nomor ini juga,
    # jadi dia gak usah nyalin kode 8 digit yang gampang ketuker/salah digit.
    daftar = "\n".join(f'{i+1}. {k["description"]}' for i, k in enumerate(kandidat))
    prompt = f"""Produk yang dicari: "{query}"

Kandidat (masing-masing punya NOMOR):
{daftar}

Urutkan kandidat di atas dari PALING cocok ke paling tidak cocok untuk produk tersebut.
Pikirkan satu per satu, dan ikuti urutan prioritas ini untuk SEMUA jenis barang:

1. JENIS barang harus benar-benar sama dengan produk yang dicari. Kandidat untuk barang
   lain yang cuma satu kelompok, satu famili, atau sekadar mirip itu TIDAK cocok,
   walaupun deskripsinya panjang dan detail. Cek bendanya dulu, bukan panjang teksnya.
2. Setelah jenisnya cocok, baru cocokkan KONDISI atau bentuknya (mis. hidup/segar/beku/
   kering/olahan untuk makanan; baru/bekas; rakitan/terurai; dsb).
3. Kalau jenis dan kondisi sama-sama cocok, BARU pilih yang paling spesifik sesuai
   spesifikasi teknis (ukuran, kapasitas, daya, material, mekanisme kerja).

Aturan ini berlaku universal, contoh di berbagai kategori:
- "gurita" -> octopus, BUKAN cumi/sotong (squid/cuttlefish) walau satu famili.
- "saklar MCB 1 fasa" -> circuit breaker, BUKAN saklar biasa atau relay.
- "obeng" -> screwdriver, BUKAN kunci pas atau perkakas tangan lain.
- "ban motor" -> ban kendaraan bermotor, BUKAN ban sepeda atau ban dalam.

Intinya: kalau jenis barangnya beda, taruh di bawah, JANGAN dinaikkan cuma karena
deskripsinya kelihatan lebih lengkap/detail.

PENTING soal alasan yang kamu tulis -- tulis JUJUR sesuai kenyataan:
- Kalau kandidat benar-benar cocok, bilang cocok dan sebutkan kenapa.
- Kalau kandidat cuma PALING MENDEKATI tapi tidak sepenuhnya pas (beda konteks,
  kondisi, atau jenis), katakan terus terang. Contoh: "paling mendekati yang ada,
  tapi ini untuk dapur sedangkan produknya untuk ruang makan".
- JANGAN memaksakan bilang "cocok" untuk kandidat yang jelas beda konteks/jenis.

Jawab dalam format JSON array, urut dari paling cocok ke paling tidak cocok.
Sebut kandidat pakai NOMOR-nya (bukan kode), plus alasan singkat (1 kalimat):
[
  {{"nomor": 1, "alasan": "..."}},
  {{"nomor": 5, "alasan": "..."}}
]
Jawab HANYA JSON, tanpa teks lain."""

    teks = None
    for i in range(retries):   # coba maksimal 3x kalau groq gagal
        try:
            resp = _client(api_key).chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            teks = resp.choices[0].message.content
            break
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 * (i + 1))   # backoff: 2s, 4s
            else:
                # balikin kandidat urutan embedding apa adanya biar app gak mati
                print(f"[WARN] rerank GAGAL (rate limit?): {str(e)[:70]} -> urutan embedding apa adanya")
                for k in kandidat:
                    k["alasan"] = ""
                return kandidat

    # olah jawaban llm: map NOMOR -> kandidat, ambil alasannya
    urut = []
    sudah = set()

    data = _ambil_json(teks)
    if data:
        for item in data:
            # ambil nomornya, ubah ke index (nomor 1 = kandidat ke-0)
            try:
                idx = int(item.get("nomor")) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(kandidat):
                k = kandidat[idx]
                if k["hs_code"] not in sudah:
                    k["alasan"] = str(item.get("alasan", "")).strip()
                    urut.append(k)
                    sudah.add(k["hs_code"])
    else:
        # json gagal di-parse -> fallback: urutan embedding apa adanya (tanpa alasan)
        print("[WARN] JSON rerank gagal di-parse -> fallback urutan embedding")

    # kandidat yang llm lupa sebutin (atau semua, kalau fallback) -> taro di belakang
    for k in kandidat:
        if k["hs_code"] not in sudah:
            k["alasan"] = ""
            urut.append(k)
            sudah.add(k["hs_code"])
    return urut