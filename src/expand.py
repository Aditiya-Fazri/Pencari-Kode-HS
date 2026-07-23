import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _client(api_key=None):
    # koneksi ke groq. pakai api_key dari argumen (diinput user di streamlit) kalau ada,
    # kalau kosong ambil dari .env (buat jalan lokal / lewat api).
    return Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))


def expand_query(query, api_key=None, retries=3):
    # query user diselipin ke dalam prompt ({query}), terus dikirim ke llm
    prompt = f'''Product: "{query}"

You are a customs tariff classification expert with deep knowledge of the Harmonized System (HS) nomenclature. Your task is to identify this product and describe it using official HS tariff terminology.

Step 1 - IDENTIFY: Use brand + model number as clues. Do NOT treat model numbers as literal specs. Identify the EXACT product variant (not just the brand's most famous product line).

Step 2 - DESCRIBE: Write a short description using HS tariff language:
- Circuit breakers: specify "moulded case" or "non-moulded", include ampere rating and poles
- Relays/contactors: always use "relay" not "contactor", include voltage and poles  
- Knives: specify "fixed blade" or "folding blade" and use (boning, hunting, kitchen, etc)
- Bearings: specify type (ball, roller, needle, tapered)
- Plugs: use "sparking plug" not "spark plug"

Output: MAX 10 words, no brand names, no model numbers, nothing else.'''

    for i in range(retries):   # coba maksimal 3x kalau groq gagal
        try:
            resp = _client(api_key).chat.completions.create(
                model="openai/gpt-oss-120b", temperature=0,   # temp 0 = jawaban konsisten
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()        # ambil teks jawaban llm
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 * (i + 1))   # backoff: tunggu 2s, 4s sebelum coba lagi
            else:
                # balikin query asli biar app gak mati (cuma hasil kurang optimal)
                print(f"[WARN] expand GAGAL (rate limit?): {str(e)[:70]} -> pakai query asli")
                return query
