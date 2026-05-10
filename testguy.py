import json
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# --- AYARLAR ---
# LM Studio'da tüm ayarları yaptığımız için burası sadeleşti
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

INPUT_FILE = "human-labeled-sample-1495.csv"
OUTPUT_FILE = "gemma-3-27B-it-QAT-Q4_0_sonuclar.json"
SUTUN_ADI = "text"  # Senin CSV sütun başlığın

# --- 1. VERİ OKUMA ---
try:
    df = pd.read_csv(INPUT_FILE, sep=";")


    # Sütun adı kontrolü (Garanti olsun)
    if SUTUN_ADI not in df.columns:
        print(f"HATA: '{SUTUN_ADI}' sütunu bulunamadı! Mevcut sütunlar: {list(df.columns)}")
        # Belki 'Text' büyük harflidir diye alternatif kontrol
        if "Text" in df.columns:
            SUTUN_ADI = "Text"
            print("-> 'Text' sütunu bulundu, onunla devam ediliyor.")
        else:
            exit()

    # Eğer dosya çok büyükse ve sample alıyorsan, random_state SABİT olmalı ki
    # programı yeniden başlattığında yine aynı satırlar gelsin.
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=42).reset_index(drop=True)

    tum_yorumlar = df[SUTUN_ADI].astype(str).tolist()
    print(f"Hedef: Toplam {len(tum_yorumlar)} yorum analiz edilecek.")

except Exception as e:
    print(f"CSV Hatası: {e}")
    exit()

# --- 2. KALDIĞIMIZ YERİ BULMA (CHECKPOINT) ---
mevcut_sonuclar = []

if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            mevcut_sonuclar = json.load(f)
        print(f"✅ Önceki kayıt bulundu! {len(mevcut_sonuclar)} tanesi zaten yapılmış.")
    except:
        print("⚠️ Kayıt dosyası bozuk veya boş, sıfırdan başlanıyor.")
        mevcut_sonuclar = []

baslangic_index = len(mevcut_sonuclar)

# Eğer hepsi bitmişse boşuna yorma
if baslangic_index >= len(tum_yorumlar):
    print("🎉 Tüm analizler zaten tamamlanmış! Dosya hazır.")
    exit()

print(f"🚀 {baslangic_index + 1}. yorumdan devam ediliyor...")

# --- 3. DÖNGÜ VE KAYDETME ---
# tqdm'e initial parametresini veriyoruz ki bar doğru yerden başlasın
for i in tqdm(range(baslangic_index, len(tum_yorumlar)), initial=baslangic_index, total=len(tum_yorumlar)):
    yorum = tum_yorumlar[i]

    try:
        # System Prompt'u LM Studio arayüzünden ayarladık, burası boş kalabilir
        # Veya garanti olsun diye basit bir reminder atabiliriz.
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "user", "content": f'Yorum: "{yorum}"'}
            ],
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()

        # Structured Output kullansan bile bazen temizlik gerekebilir
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(clean_content)

        mevcut_sonuclar.append({
            "index": i,
            "yorum": yorum,
            "analiz": parsed_json
        })

    except Exception as e:
        # Hata olursa da kaydet, durmasın
        mevcut_sonuclar.append({
            "index": i,
            "yorum": yorum,
            "error": str(e)
        })

    # --- KRİTİK KISIM: HER 100 ADETTE BİR KAYDET ---
    if (i + 1) % 100 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(mevcut_sonuclar, f, ensure_ascii=False, indent=4)
        # Tqdm barını bozmamak için print yapmıyoruz, arkada kaydetti.

# --- BİTİŞTE SON KAYIT ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(mevcut_sonuclar, f, ensure_ascii=False, indent=4)

print(f"\n🏁 İŞLEM TAMAMLANDI! Toplam {len(mevcut_sonuclar)} sonuç kaydedildi.")