import json
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

SRC = "data/CategorizedGeneralQuestions.json"
DST = "data/cog_rag_documents.json"

TITLES = {
    "bolum_dersleri": "KTÜN BM Ders Programı ve Ders Bilgileri",
    "staj":           "KTÜN BM Staj Süreci ve Kuralları",
    "obs":            "KTÜN OBS Öğrenci Bilgi Sistemi Kullanımı",
    "yataygecis":     "KTÜN Yatay Geçiş Başvuru ve Şartları",
    "yaz_okulu":      "KTÜN Yaz Okulu Bilgileri",
    "erasmus":        "KTÜN Erasmus Programı",
    "yemekhane":      "KTÜN Yemekhane Bilgileri",
    "lms":            "KTÜN LMS Öğrenme Yönetim Sistemi",
    "genel":          "KTÜN Genel Akademik Bilgiler",
    "konu_disi":      "Kapsam Dışı Konular",
}

LABELS = {
    "bolum_dersleri": "[DERS BİLGİSİ]",
    "staj":           "[STAJ]",
    "obs":            "[OBS]",
    "yataygecis":     "[YATAY GEÇİŞ]",
    "yaz_okulu":      "[YAZ OKULU]",
    "erasmus":        "[ERASMUS]",
    "yemekhane":      "[YEMEKHANE]",
    "lms":            "[LMS]",
    "genel":          "[GENEL]",
    "konu_disi":      "[KAPSAM DIŞI]",
}

# Türkçe stopword listesi
STOPWORDS = {
    "ve", "veya", "ile", "bu", "bir", "için", "de", "da", "den", "dan",
    "in", "nin", "nın", "nun", "nün", "ta", "te", "ki", "mi", "mu", "mü",
    "mı", "ne", "ya", "hem", "ama", "fakat", "ancak", "çünkü", "eğer",
    "olan", "olur", "olarak", "olan", "olan", "gibi", "kadar", "daha",
    "her", "hiç", "bazı", "tüm", "tüm", "birçok", "pek", "çok", "az",
    "en", "nasıl", "neden", "nerede", "hangi", "hangi", "nereye", "nereden",
    "var", "yok", "dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür",
    "olan", "olan", "olması", "olup", "olduğu", "olduğunda", "olduğu",
    "ile", "veya", "ya", "de", "da", "ki", "ise", "hem", "ne", "o", "bu",
    "şu", "ben", "sen", "biz", "siz", "onlar", "onun", "bunun", "şunun",
    "can", "alır", "alınır", "yapılır", "edilir", "verilir", "gider",
    "gerekir", "olmalı", "olabilir", "yapabilir", "aldığı", "verilen",
}

def extract_keywords(text: str, n: int = 5) -> list[str]:
    """Metinden n adet anahtar kelime çıkarır."""
    # Noktalama temizle, küçük harf
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()
    # Stopword ve kısa kelimeleri filtrele
    candidates = [
        w for w in words
        if len(w) > 3 and w not in STOPWORDS and not w.isdigit()
    ]
    # Frekans sırala, benzersiz tut
    freq: dict[str, int] = {}
    for w in candidates:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda x: -freq[x])
    return ranked[:n]


# ─── Yükle ────────────────────────────────────────────────────────────────────
with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

# ─── Kategoriye göre grupla ───────────────────────────────────────────────────
cat_rows: dict[str, list] = defaultdict(list)
for row in data:
    cat_rows[row["category"]].append(row)

# ─── Her kategori için doküman oluştur ────────────────────────────────────────
documents = []

for cat in TITLES:
    rows = cat_rows.get(cat, [])
    if not rows:
        continue

    label  = LABELS[cat]
    title  = TITLES[cat]
    source = rows[0].get("source", "")

    # Benzersiz cevapları topla (görünme sırası korunur)
    seen_outputs: set[str] = set()
    unique_outputs: list[str] = []
    for row in rows:
        out = row.get("output", "").strip()
        if out and out not in seen_outputs:
            seen_outputs.add(out)
            unique_outputs.append(out)

    # Her benzersiz cevap için bir blok oluştur
    blocks: list[str] = []
    for output in unique_outputs:
        keywords = extract_keywords(output)
        kw_str   = ", ".join(keywords) if keywords else "—"
        block    = f"{label} [anahtar: {kw_str}]\n{output}"
        blocks.append(block)

    content = "\n\n".join(blocks)

    documents.append({
        "doc_id":       f"{cat}_doc",
        "theme":        cat,
        "title":        title,
        "content":      content,
        "source":       source,
        "kayit_sayisi": len(rows),
        "blok_sayisi":  len(blocks),
    })

# ─── Kaydet ───────────────────────────────────────────────────────────────────
with open(DST, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

# ─── Rapor ────────────────────────────────────────────────────────────────────
docs_sorted = sorted(documents, key=lambda d: len(d["content"]), reverse=True)
total_chars = sum(len(d["content"]) for d in documents)

print(f"{'='*65}")
print(f"COG-RAG DOKÜMANLARI (YENİ FORMAT) → {DST}")
print(f"{'='*65}")
print(f"Toplam doküman : {len(documents)}")
print(f"Toplam içerik  : {total_chars:,} karakter ({total_chars/1024/1024:.2f} MB)\n")

print(f"  {'Doküman':<25} {'Kayıt':>6}  {'Blok':>5}  {'Karakter':>10}  {'KB':>6}")
print(f"  {'-'*58}")
for doc in sorted(documents, key=lambda d: d["kayit_sayisi"], reverse=True):
    chars = len(doc["content"])
    print(f"  {doc['doc_id']:<25} {doc['kayit_sayisi']:>6}  "
          f"{doc['blok_sayisi']:>5}  {chars:>10,}  {chars/1024:>5.1f}")

print(f"\n  En uzun : {docs_sorted[0]['doc_id']}  ({len(docs_sorted[0]['content']):,} kar)")
print(f"  En kısa : {docs_sorted[-1]['doc_id']}  ({len(docs_sorted[-1]['content']):,} kar)")

# İlk 3 bloğun önizlemesi — her doküman için
print(f"\n{'='*65}")
print("İLK 3 BLOK ÖNİZLEMESİ (her doküman)")
print(f"{'='*65}")
for doc in documents:
    blocks_preview = doc["content"].split("\n\n")[:3]
    print(f"\n── {doc['doc_id']} ({doc['blok_sayisi']} blok) ──")
    for i, blk in enumerate(blocks_preview, 1):
        first_line = blk.split("\n")[0]        # etiket satırı
        second_line = blk.split("\n")[1][:90] if "\n" in blk else ""  # cevap başı
        print(f"  [{i}] {first_line}")
        if second_line:
            print(f"       {second_line}...")
print(f"\n{'='*65}")
