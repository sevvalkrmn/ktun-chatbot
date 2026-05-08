from dotenv import load_dotenv

load_dotenv()

# ── LLM (Ollama) ─────────────────────────────────────────
# Jetson'da: ollama pull gemma4:27b
# Modelin tam adını `ollama list` ile doğrula
OLLAMA_MODEL = "gemma4:27b"
OLLAMA_HOST  = "http://localhost:11434"   # Jetson üzerinde çalışan Ollama

# ── Veri Seti ────────────────────────────────────────────
DATA_PATH = "data/ktun_dataset_v2.json"

CATEGORY_FIELD = "category"

# ── RAG Eşik Değerleri ───────────────────────────────────
HIGH_CONFIDENCE  = 0.70   # Direkt veri setinden cevap dön
MID_CONFIDENCE   = 0.50  # LLM'e bağlam ver, sentez yaptır
LOW_CONFIDENCE   = 0.10   # Konu dışı, reddet

# ── Hibrit Arama Ağırlıkları ─────────────────────────────
BM25_WEIGHT      = 0.40
SEMANTIC_WEIGHT  = 0.60

# ── Embedding Modeli ─────────────────────────────────────
EMBEDDING_MODEL  = "BAAI/bge-m3"

# ── Retrieval ────────────────────────────────────────────
TOP_K            = 5      # Kaç sonuç getirilsin

# ── Sistem Promptu ───────────────────────────────────────
SYSTEM_PROMPT = """Sen KTÜN Bilgisayar Mühendisliği Bölümü Akademik Bilgi Asistanısın.
Görevin öğrencilerin bölümle ilgili sorularını yanıtlamaktır.
Sana verilen BAĞLAM bilgisine dayanarak cevap ver.
Bağlamda olmayan bir bilgiyi kesinlikle uydurma.
Bilmediğin konularda 'Bu konuda kesin bilgim bulunmuyor, bölüm sekreterliğine danışabilirsiniz.' de.
Sadece Türkçe cevap ver."""