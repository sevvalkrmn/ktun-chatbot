import os
import sys
import json
import re
import fitz  # pymupdf

sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR      = "dökümanlar"
OUTPUT        = "data/ktun_dataset_v2.json"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

CATEGORY_MAP = {
    "erasmus":          ["Erasmus", "erasmus"],
    "yatay_gecis":      ["yatay", "Yatay", "YATAY"],
    "staj":             ["staj", "Staj"],
    "ders":             ["Ders", "ders", "DERS", "program", "Program", "Öğretim_Planı", "retim_Plan"],
    "sinav":            ["Sınav", "sinav"],
    "yaz_okulu":        ["Yaz", "yaz"],
    "duyuru":           ["Duyuru", "duyuru"],
    "hakkimizda":       ["Hakkımızda", "hakkimizda", "Tanitim", "tanitim"],
    "akademik_personel":["AkademikPersonel", "Akademik"],
    "komisyon":         ["komisyon", "Komisyon"],
    "bitirme":          ["Bitirme", "bitirme"],
}

def get_category(filename: str) -> str:
    for category, keywords in CATEGORY_MAP.items():
        if any(kw in filename for kw in keywords):
            return category
    return "genel"

def extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_docx(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    for table in doc.tables:
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                if all(h for h in headers):
                    row_text = ", ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c)
                else:
                    row_text = " | ".join(c for c in cells if c)
                if row_text.strip():
                    parts.append(row_text)
    return "\n".join(parts)

def extract_bolum_dersleri(path: str) -> list:
    """
    BölümDersleri.docx için dönem bazlı özel işleme.
    XML body elementlerini sırayla okuyarak her dönem → ayrı chunk.
    """
    from docx.text.paragraph import Paragraph
    from docx.table import Table as DocxTable

    doc = Document(path)
    donem_data = {}
    current_donem = None

    for child in doc.element.body:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

        if tag == 'p':
            p = Paragraph(child, doc)
            text = p.text.strip()
            match = re.search(r'DÖNEM\s*(\d+)', text, re.IGNORECASE)
            if match:
                current_donem = f"DÖNEM {match.group(1)}"
                if current_donem not in donem_data:
                    donem_data[current_donem] = []

        elif tag == 'tbl' and current_donem:
            table = DocxTable(child, doc)
            headers = [cell.text.strip() for cell in table.rows[0].cells]
            for row in table.rows[1:]:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    if all(h for h in headers):
                        row_text = ", ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c)
                    else:
                        row_text = " | ".join(c for c in cells if c)
                    if row_text.strip():
                        donem_data[current_donem].append(row_text)

    results = []
    for donem, dersler in donem_data.items():
        if dersler:
            content = f"BİLGİSAYAR MÜHENDİSLİĞİ {donem} DERSLERİ:\n" + "\n".join(dersler)
            results.append((donem, content))

    return results

def extract_ogretim_plani(path: str) -> list:
    """
    2025-2026_Öğretim_Planı.pdf için özel işleyici.
    Yarıyıl + sınıf bazlı ders listelerini ve bağlı ders (ön koşul) bilgilerini üretir.
    """
    doc = fitz.open(path)
    pages = [doc[i].get_text() for i in range(len(doc))]
    doc.close()

    # ── Sayfa → yarıyıl eşleşmesi (PDF yapısından okundu) ────────────────
    # Her sayfa hangi sınıf / yarıyılları kapsıyor
    YARIL_SINIF = [
        ("1. Yarıyıl (Güz)",    "1. SINIF, 1. Yarıyıl"),
        ("2. Yarıyıl (Bahar)",  "1. SINIF, 2. Yarıyıl"),
        ("3. Yarıyıl (Güz)",    "2. SINIF, 3. Yarıyıl"),
        ("4. Yarıyıl (Bahar)",  "2. SINIF, 4. Yarıyıl"),
        ("5. Yarıyıl (Güz)",    "3. SINIF, 5. Yarıyıl"),
        ("6. Yarıyıl (Bahar)",  "3. SINIF, 6. Yarıyıl"),
        ("7. Yarıyıl (Güz)",    "4. SINIF, 7. Yarıyıl"),
        ("8. Yarıyıl (Bahar)",  "4. SINIF, 8. Yarıyıl"),
    ]

    # ── Ön koşul efsanesi (son sayfadan çıkarıldı) ───────────────────────
    LEGEND = {
        "[1]": (
            "Algoritma ve Programlama 1 (BIL102) veya "
            "Algoritma ve Programlama 2 (BIL201)"
        ),
        "[2]": "Lojik Tasarım (BIL303)",
        "[3]": "Diferansiyel Denklemler (BBF401)",
        "[4]": "Bilgisayar Mühendisliği Uygulaması 1 (BIL702)",
        "[5]": (
            "Bilgisayar Mühendisliği Uygulaması 1 (BIL702) dersinden başarılı "
            "VE Bilgisayar Mühendisliği Uygulaması 2 (BIL801) dersinden başarısız"
        ),
    }

    # ── PDF'deki tüm ders satırlarını çıkar ──────────────────────────────
    # Satır formatı: "KOD DersAdı" veya "KOD\nDersAdı", ardından T U Yerel AKTS [prereq] Yz
    full_text = "\n".join(pages)
    # Her satırı temizle
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    # Ders kodu regex: BBFxxx veya BILxxx (isteğe bağlı 3 rakam sonrası)
    code_re = re.compile(r'^(BBF|BIL)\d{3}$')
    code_inline_re = re.compile(r'^((BBF|BIL)\d{3})\s+(.+)$')

    courses = []  # list of (code, name, prereq_refs)
    i = 0
    while i < len(lines):
        line = lines[i]
        # "BBF101 Matematik 1" — kod ve isim aynı satırda
        m = code_inline_re.match(line)
        if m:
            code = m.group(1)
            name = m.group(3).rstrip()
            # sonraki satırlar: T U Yerel AKTS [prereq...] Yz
            refs = _collect_refs(lines, i + 1)
            courses.append((code, name, refs))
            i += 1
            continue
        # "BIL304" — tek başına kod, sonraki satır isim
        if code_re.match(line) and i + 1 < len(lines):
            code = line
            name = lines[i + 1]
            refs = _collect_refs(lines, i + 2)
            courses.append((code, name, refs))
            i += 2
            continue
        i += 1

    # ── Yarıyıl bazlı ders listesi chunk'ları ────────────────────────────
    # Kod prefix'inden yarıyılı çıkar: BBFx00 veya BILx00 → x = yarıyıl
    def code_to_yaril(code: str) -> int:
        try:
            return int(code[3]) if code[3].isdigit() else 0
        except IndexError:
            return 0

    yaril_courses: dict[int, list] = {y: [] for y in range(1, 9)}
    for code, name, refs in courses:
        y = code_to_yaril(code)
        if 1 <= y <= 8:
            yaril_courses[y].append((code, name, refs))

    records = []
    sinif_map = {1: "1. SINIF", 2: "1. SINIF", 3: "2. SINIF", 4: "2. SINIF",
                 5: "3. SINIF", 6: "3. SINIF", 7: "4. SINIF", 8: "4. SINIF"}
    donem_label = {1: "1. YARIYIL (GÜZ)", 2: "2. YARIYIL (BAHAR)",
                   3: "3. YARIYIL (GÜZ)", 4: "4. YARIYIL (BAHAR)",
                   5: "5. YARIYIL (GÜZ)", 6: "6. YARIYIL (BAHAR)",
                   7: "7. YARIYIL (GÜZ)", 8: "8. YARIYIL (BAHAR)"}

    for y in range(1, 9):
        dlist = yaril_courses[y]
        if not dlist:
            continue
        header = (
            f"BİLGİSAYAR MÜHENDİSLİĞİ {sinif_map[y]} "
            f"{donem_label[y]} DERSLERİ (ÖĞRETİM PLANI 2025-2026):\n"
        )
        lines_out = []
        for code, name, refs in dlist:
            prereq_codes = [r for r in refs if r in LEGEND]
            if prereq_codes:
                prereq_str = "; ".join(
                    f"Ön koşul: {LEGEND[r]} gereklidir" for r in prereq_codes
                )
                lines_out.append(f"{name} ({code}) — {prereq_str}")
            else:
                lines_out.append(f"{name} ({code})")
        records.append({
            "id":          f"ogretim_plani_yaril_{y}",
            "source":      os.path.basename(path),
            "category":    "ders",
            "content":     header + "\n".join(lines_out),
            "chunk_index": 0,
        })

    # ── Bağlı ders özet chunk'u ───────────────────────────────────────────
    prereq_courses = [(code, name, refs) for code, name, refs in courses
                      if any(r in LEGEND for r in refs)]
    if prereq_courses:
        lines_out = [
            "BİLGİSAYAR MÜHENDİSLİĞİ BAĞLI DERS VE ÖN KOŞUL BİLGİLERİ:\n",
            "Aşağıdaki dersler bağlı ders (ön koşul) gerektirmektedir. "
            "Bu dersleri alabilmek için belirtilen dersten başarılı olmak zorunludur:\n",
        ]
        for code, name, refs in prereq_courses:
            prereq_codes = [r for r in refs if r in LEGEND]
            for r in prereq_codes:
                cond = LEGEND[r]
                if r == "[5]":
                    lines_out.append(
                        f"- {name} ({code}): {cond} olan öğrenciler alabilir."
                    )
                else:
                    lines_out.append(
                        f"- {name} ({code}): Bu dersi alabilmek için {cond} "
                        f"dersinden başarılı olmak gereklidir."
                    )
        records.append({
            "id":          "ogretim_plani_on_kosullar",
            "source":      os.path.basename(path),
            "category":    "ders",
            "content":     "\n".join(lines_out),
            "chunk_index": 0,
        })

    # ── Sınıf ↔ yarıyıl açıklama chunk'u ─────────────────────────────────
    records.append({
        "id":       "ogretim_plani_sinif_yaril_eslesme",
        "source":   os.path.basename(path),
        "category": "ders",
        "content": (
            "BİLGİSAYAR MÜHENDİSLİĞİ SINIF VE YARIYIL EŞLEŞMESİ:\n"
            "1. sınıf: 1. yarıyıl (güz) ve 2. yarıyıl (bahar)\n"
            "2. sınıf: 3. yarıyıl (güz) ve 4. yarıyıl (bahar)\n"
            "3. sınıf: 5. yarıyıl (güz) ve 6. yarıyıl (bahar)\n"
            "4. sınıf: 7. yarıyıl (güz) ve 8. yarıyıl (bahar)"
        ),
        "chunk_index": 0,
    })

    return records


def _collect_refs(lines: list, start: int) -> list:
    """Bir ders satırının hemen ardından gelen [X] referanslarını toplar."""
    refs = []
    for line in lines[start:start + 8]:
        found = re.findall(r'\[\d\]', line)
        refs.extend(found)
        # Yz veya Uz görününce ders satırı bitti
        if line in ("Yz", "Uz"):
            break
    return refs


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\x0c', '', text)
    return text.strip()

def build_dataset():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )

    records = []
    skipped = []

    for filename in sorted(os.listdir(DOCS_DIR)):
        filepath = os.path.join(DOCS_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        source = os.path.splitext(filename)[0]
        category = get_category(filename)

        try:
            # Öğretim Planı PDF için özel işleme
            if filename == "2025-2026_Öğretim_Planı.pdf":
                plan_records = extract_ogretim_plani(filepath)
                records.extend(plan_records)
                print(f"OK {filename}: {len(plan_records)} chunk")
                continue

            # BölümDersleri.docx için özel dönem bazlı işleme
            if filename == "BölümDersleri.docx":
                donem_results = extract_bolum_dersleri(filepath)
                for donem_name, content in donem_results:
                    safe_id = re.sub(r'[^a-z0-9_]', '_', donem_name.lower())
                    records.append({
                        "id":          f"ders_{safe_id}",
                        "source":      filename,
                        "category":    "ders",
                        "content":     content,
                        "chunk_index": 0
                    })
                print(f"OK {filename}: {len(donem_results)} donem chunk")
                continue

            # Diğer dosyalar
            if ext == ".pdf":
                raw = extract_pdf(filepath)
            elif ext in (".docx", ".doc"):
                raw = extract_docx(filepath)
            else:
                skipped.append(filename)
                continue

            text = clean_text(raw)
            if not text:
                skipped.append(filename)
                continue

            chunks = splitter.split_text(text)
            safe_source = re.sub(r'[^a-z0-9_]', '_', source[:25].lower())

            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if len(chunk) < 30:
                    continue
                records.append({
                    "id":          f"{category}_{safe_source}_{i:03d}",
                    "source":      filename,
                    "category":    category,
                    "content":     chunk,
                    "chunk_index": i
                })

            print(f"OK {filename}: {len(chunks)} chunk")

        except Exception as e:
            print(f"HATA {filename}: {e}")
            skipped.append(filename)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nToplam: {len(records)} chunk → {OUTPUT}")
    if skipped:
        print(f"Atlanan: {skipped}")

if __name__ == "__main__":
    build_dataset()
