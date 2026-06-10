"""
Yeni dökümanlardan (ders_hocalari.docx, staj_süreçleri.docx + gömülü görseller,
2025-2026_Öğretim_Planı.pdf) çıkarılan bilgileri knowledge_base'e ekler.

Çalıştırdıktan sonra: python build_kb_dataset.py
"""
import os
import re
import sys
import glob

sys.stdout.reconfigure(encoding="utf-8")
from docx import Document

KB        = os.path.join("data", "knowledge_base")
DERS_DIR  = os.path.join(KB, "bolum_dersleri")
STAJ_DIR  = os.path.join(KB, "staj")
HOCA_DOCX = os.path.join("data", "ders_hocalari.docx")

DONEM_TR = {1: "1. Dönem", 2: "2. Dönem", 3: "3. Dönem", 4: "4. Dönem",
            5: "5. Dönem", 6: "6. Dönem", 7: "7. Dönem", 8: "8. Dönem"}

# knowledge_base ders adı (norm) → ders_hocalari.docx'teki ders adı.
# İki kaynakta farklı adlandırılan dersleri elle eşler.
ALIAS = {
    "veri tabani i": "Veritabanı Yönetim Sistemleri",
    "yazilim muhendisligi": "Yazılım Mühendisliğine Giriş",
    "bilgisayar muhendisligi uygulamasi i bitirme projesi i":
        "Bilgisayar Mühendisliği Uygulaması I",
}


# ── İsim normalleştirme (eşleştirme için) ───────────────────────────────
def norm(name: str) -> str:
    name = name.lower()
    for a, b in {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c", "â": "a"}.items():
        name = name.replace(a, b)
    name = re.sub(r"\((tsd|tosd)\)", "", name)      # seçmeli etiketleri
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ── ders_hocalari.docx ayrıştır ─────────────────────────────────────────
def parse_hocalar() -> list:
    doc = Document(HOCA_DOCX)
    records = []
    for ti, table in enumerate(doc.tables, start=1):
        donem = ti  # Tablo sırası dönem numarasıdır (DÖNEM 1..8)
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) < 4:
                continue
            code, name, akts, instructor = cells[0], cells[1], cells[2], cells[3]
            if not code or code.upper().startswith("DÖNEM") or code == "Ders Kodu":
                continue
            if not instructor:
                continue
            records.append({
                "donem": donem, "code": code, "name": name,
                "akts": akts, "instructor": instructor,
            })
    return records


# ── Her ders .md dosyasına koordinatör ekle ─────────────────────────────
def inject_instructors(records: list):
    by_name = {norm(r["name"]): r for r in records}
    matched, unmatched = [], []

    for path in glob.glob(os.path.join(DERS_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if "Course Instructor" in text:        # zaten eklenmiş
            continue

        m = re.search(r"#\s*Topic:\s*(.+)", text)
        if not m:
            continue
        topic = m.group(1).strip()
        topic_name = re.sub(r"\s*\([^)]*\)\s*$", "", topic)   # sondaki (kod) at
        key = norm(topic_name)
        rec = by_name.get(key) or by_name.get(norm(ALIAS.get(key, "")))

        if not rec:
            unmatched.append(topic_name)
            continue

        section = (
            f"\n## Course Instructor\n"
            f"{rec['name']} dersinin koordinatörü/sorumlu öğretim üyesi "
            f"{rec['instructor']}'dir. Ders {DONEM_TR[rec['donem']]}'inde verilir "
            f"({rec['code']}, {rec['akts']} AKTS).\n"
            f"- {topic_name} dersine hangi hoca giriyor?\n"
            f"- {topic_name} dersinin sorumlusu kim?\n"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(section)
        matched.append(topic_name)

    return matched, unmatched


# ── Staj görsellerinden çıkarılan yeni dosyalar ─────────────────────────
def write_staj_docs():
    # image2.png — başvuru takvimi + "1 hafta sonra" kuralı
    basvuru = """# Topic: Staj Başvuru Takvimi ve Başlama Süresi (2026)

## Metadata
- **Category:** staj
- **Keywords:** 1 hafta, başlama tarihi, başvuru takvimi, sonuçların ilanı, staj ne zaman başlar, 2026
- **Sources:** ktun_bm_staj_basvuru_takvimi_2026

## Application Process
2026 yılı staj başvuru sonuçları her ay 15'inde ilan edilir. ÖNEMLİ KURAL: Öğrenci, staj başvuru sonuçlarının ilan edilmesinden EN ERKEN 1 HAFTA SONRA staja başlayabilir. Örneğin sonuçlar 15.01.2026 tarihinde açıklandıysa staja en erken 22.01.2026 tarihinde başlanabilir.

2026 yılı staj başvuru tarih aralıkları ve sonuçların ilan tarihleri:
- 05.01.2026 - 09.01.2026 başvurusu → sonuç ilanı 15.01.2026
- 02.02.2026 - 06.02.2026 başvurusu → sonuç ilanı 15.02.2026
- 02.03.2026 - 06.03.2026 başvurusu → sonuç ilanı 15.03.2026
- 06.04.2026 - 10.04.2026 başvurusu → sonuç ilanı 15.04.2026
- 04.05.2026 - 08.05.2026 başvurusu → sonuç ilanı 15.05.2026
- 01.06.2026 - 05.06.2026 başvurusu → sonuç ilanı 15.06.2026
- 29.06.2026 - 03.07.2026 başvurusu → sonuç ilanı 15.07.2026
- 03.08.2026 - 07.08.2026 başvurusu → sonuç ilanı 15.08.2026
- 07.09.2026 - 11.09.2026 başvurusu → sonuç ilanı 15.09.2026
- 05.10.2026 - 09.10.2026 başvurusu → sonuç ilanı 15.10.2026
- 02.11.2026 - 06.11.2026 başvurusu → sonuç ilanı 15.11.2026
- 07.12.2026 - 11.12.2026 başvurusu → sonuç ilanı 15.12.2026

## Search Terms / Frequently Asked Questions
- Staj kabul aldıktan kaç gün sonra başlayabilirim?
- Sonuçlar açıklandıktan sonra en erken ne zaman staja başlarım?
- Staj başvuru sonuçları ne zaman açıklanır?
- Ayın 15'inde sonuç açıklandı, ne zaman başlayabilirim?
- 2026 staj başvuru tarihleri neler?
"""

    # image3.png — defter teslim tarihleri
    teslim = """# Topic: Staj Defteri Teslim Tarihleri (2026)

## Metadata
- **Category:** staj
- **Keywords:** 2026, ay aralığı, defter teslim, staj defteri teslim tarihi, teslim takvimi
- **Sources:** ktun_bm_staj_defteri_teslim_2026

## Required Documents
2026 yılı staj defteri teslim tarihleri (aylara göre ay içindeki gün aralığı):
- Ocak: 26-30
- Şubat: 23-27
- Mart: 23-27
- Nisan: 27-30
- Mayıs: 18-22
- Haziran: 22-26
- Temmuz: 27-31
- Ağustos: 24-28
- Eylül: 21-30
- Ekim: 26-30
- Kasım: 23-27
- Aralık: 28-31

Staj defteri ve sicil fişi bu tarihlerde bölüm başkanlığına teslim edilmelidir. Güncel duyurular için bm.ktun.edu.tr takip edilmelidir.

## Search Terms / Frequently Asked Questions
- Staj defterini hangi tarihte teslim etmeliyim?
- 2026 staj defteri teslim tarihleri neler?
- Ağustos ayında defter ne zaman teslim edilir?
"""

    # Classroom kodu (staj_süreçleri.docx paragraf 1)
    classroom = """# Topic: Mesleki Staj Google Classroom

## Metadata
- **Category:** staj
- **Keywords:** classroom, duyuru, google classroom, mesleki staj dersi, sınıf kodu
- **Sources:** ktun_bm_staj_classroom_2025

## Internship Process
Mesleki Staj dersine ait duyurular ve dokümanlar Google Classroom üzerinden paylaşılır. Mesleki Staj Dersi Google Classroom sınıf kodu: gxw5yfk. Öğrenciler staj süreçlerini takip etmek için bu sınıfa katılmalıdır.

## Search Terms / Frequently Asked Questions
- Staj Classroom kodu nedir?
- Mesleki staj dersi sınıf kodu kaç?
- Staj duyurularını nereden takip ederim?
"""

    files = {
        "Staj_Basvuru_Takvimi_2026.md": basvuru,
        "Staj_Defteri_Teslim_Tarihleri_2026.md": teslim,
        "Mesleki_Staj_Classroom.md": classroom,
    }
    for fname, content in files.items():
        with open(os.path.join(STAJ_DIR, fname), "w", encoding="utf-8") as f:
            f.write(content)
    return list(files.keys())


def main():
    records = parse_hocalar()
    print(f"ders_hocalari.docx: {len(records)} ders-koordinatör kaydı ayrıştırıldı.")

    matched, unmatched = inject_instructors(records)
    print(f"Koordinatör eklenen ders dosyası: {len(matched)}")
    print(f"İsim eşleşmeyen ders dosyası (docx'te koordinatör verisi yok): {len(unmatched)}")
    if unmatched:
        print("  Eşleşmeyenler:", ", ".join(sorted(unmatched)))

    staj_files = write_staj_docs()
    print(f"Yeni staj dosyaları: {', '.join(staj_files)}")


if __name__ == "__main__":
    main()
