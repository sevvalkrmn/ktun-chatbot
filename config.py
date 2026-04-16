import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ── Model ────────────────────────────────────────────────
def get_model_name():
    """
    VRAM'e göre otomatik model seçimi.
    Qwen3-8B Q4'te ~5GB VRAM — 8GB GPU'da çalışır.
    Qwen2.5-7B Q4'te ~4.5GB VRAM — fallback.
    """
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 7.5:
            return "Qwen/Qwen3-8B"               # RTX 4060 8GB için ideal
        else:
            return "Qwen/Qwen2.5-7B-Instruct"   # 6GB ve altı için
    return "Qwen/Qwen2.5-7B-Instruct"           # CPU fallback

MODEL_NAME = get_model_name()

DEVICE     = "cuda"   # GPU kullan, CPU istersen "cpu" yaz

# 4-bit kuantizasyon — 7B modeli ~4.5GB VRAM'e sığdırır
LOAD_IN_4BIT = True

# ── Veri Seti ────────────────────────────────────────────
DATA_PATH = "data/ktun_dataset.json"

# Yeni veri seti alan adları (v2 format)
QUESTION_FIELD = "canonical_question"
ANSWER_FIELD   = "answer_full"
CATEGORY_FIELD = "category"

# ── RAG Eşik Değerleri ───────────────────────────────────
HIGH_CONFIDENCE  = 0.70   # Direkt veri setinden cevap dön
MID_CONFIDENCE   = 0.25   # LLM'e bağlam ver, sentez yaptır
LOW_CONFIDENCE   = 0.10   # Konu dışı, reddet

# ── Hibrit Arama Ağırlıkları ─────────────────────────────
BM25_WEIGHT      = 0.40
SEMANTIC_WEIGHT  = 0.60

# ── Embedding Modeli ─────────────────────────────────────
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Retrieval ────────────────────────────────────────────
TOP_K            = 5      # Kaç sonuç getirilsin

# ── Sistem Promptu ───────────────────────────────────────
SYSTEM_PROMPT = """Sen KTÜN Bilgisayar Mühendisliği Bölümü Akademik Bilgi Asistanısın.
Görevin öğrencilerin bölümle ilgili sorularını yanıtlamaktır.
Sana verilen BAĞLAM bilgisine dayanarak cevap ver.
Bağlamda olmayan bir bilgiyi kesinlikle uydurma.
Bilmediğin konularda 'Bu konuda kesin bilgim bulunmuyor, bölüm sekreterliğine danışabilirsiniz.' de.
Sadece Türkçe cevap ver."""