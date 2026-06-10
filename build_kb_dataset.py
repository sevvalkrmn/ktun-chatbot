import os
import sys
import json
import re

sys.stdout.reconfigure(encoding="utf-8")

# ── Ayarlar ──────────────────────────────────────────────────────────────
KB_DIR        = os.path.join("data", "knowledge_base")
OUTPUT        = os.path.join("data", "ktun_kb_dataset.json")
MAX_CHARS     = 1200   # Bu uzunluğun üstündeki dosyalar parçalara bölünür
OVERLAP_CHARS = 120


def slugify(text: str) -> str:
    text = text.lower()
    repl = {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c"}
    for k, v in repl.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def clean_md(text: str) -> str:
    """Markdown gürültüsünü hafifçe temizler, metni korur."""
    # Başlık işaretlerini ('#', '-', '*') sadeleştir ama içeriği koru
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # "## Course Objectives" gibi başlıklardaki # işaretlerini kaldır
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        # Liste işaretlerini kaldır
        stripped = re.sub(r"^[-*]\s*", "", stripped)
        # Bold/italic markdown işaretlerini kaldır
        stripped = stripped.replace("**", "").replace("__", "")
        lines.append(stripped)
    return "\n".join(lines)


def split_text(text: str, max_chars: int, overlap: int) -> list:
    """Uzun metni paragraf sınırlarına saygı göstererek parçalar."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n")
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            # Overlap: önceki parçanın sonundan bir miktar taşı
            buf = buf[-overlap:] + "\n" + para
        else:
            buf = (buf + "\n" + para) if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def build():
    records = []
    file_count = 0

    for root, _, files in os.walk(KB_DIR):
        for filename in sorted(files):
            if not filename.lower().endswith(".md"):
                continue

            filepath = os.path.join(root, filename)
            # Kategori = knowledge_base altındaki klasör adı
            category = os.path.basename(root)
            source   = os.path.splitext(filename)[0]

            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()

            text = clean_md(raw)
            if not text:
                continue

            file_count += 1
            chunks = split_text(text, MAX_CHARS, OVERLAP_CHARS)
            base_id = slugify(f"{category}_{source}")

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue
                records.append({
                    "id":          f"{base_id}_{i:03d}",
                    "source":      filename,
                    "category":    category,
                    "content":     chunk.strip(),
                    "chunk_index": i,
                })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"{file_count} markdown dosyası işlendi.")
    print(f"Toplam {len(records)} chunk → {OUTPUT}")


if __name__ == "__main__":
    build()
