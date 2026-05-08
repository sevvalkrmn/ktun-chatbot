import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from config import (
    DATA_PATH, EMBEDDING_MODEL,
    BM25_WEIGHT, SEMANTIC_WEIGHT, TOP_K,
    HIGH_CONFIDENCE, MID_CONFIDENCE, LOW_CONFIDENCE
)

# ── VERİ YÜKLEMESİ ──────────────────────────────────────────────────────
def load_data(path: str) -> list:
    """JSON veri setini yükler."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── CORPUS OLUŞTURMA ─────────────────────────────────────────────────────
def build_corpus(data: list) -> tuple:
    """
    Döküman chunk formatını (content alanı) okur.
    corpus_texts  → BM25 ve FAISS için index metni
    corpus_answers → kullanıcıya döndürülecek içerik
    """
    corpus_texts   = []
    corpus_answers = []
    seen_ids       = set()

    for item in data:
        id_     = item.get("id", "")
        content = item.get("content", "")

        if not content:
            continue
        if id_ and id_ in seen_ids:
            continue
        if id_:
            seen_ids.add(id_)

        corpus_texts.append(content)
        corpus_answers.append(content)

    return corpus_texts, corpus_answers

# ── BM25 İNDEKSİ ────────────────────────────────────────────────────────
def build_bm25(corpus_texts: list) -> BM25Okapi:
    """Tokenize edilmiş corpus üzerinde BM25 indeksi oluşturur."""
    tokenized = [text.lower().split() for text in corpus_texts]
    return BM25Okapi(tokenized)

# ── FAISS İNDEKSİ ───────────────────────────────────────────────────────
def build_faiss(corpus_texts: list, model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Corpus metinlerini vektöre çevirip FAISS indeksi oluşturur."""
    embeddings = model.encode(corpus_texts, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )  # normalize — iç çarpım = cosine benzerliği

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index

# ── BM25 SORGU ÖN İŞLEME ────────────────────────────────────────────────
_STOP_WORDS = {
    # Soru kalıpları
    "hakkında", "bilgi", "ver", "verir", "misin", "miain", "nedir", "nelerdir",
    "nasıl", "anlat", "açıkla", "söyle", "anlatabilir",
    "öğrenmek", "istiyorum", "merak", "ediyorum",
    "lütfen", "mısın", "misiniz", "musun",
    # Kişisel bağlam
    "ben", "bana", "benim", "biz", "bizim", "siz", "sizin",
    "öğrencisiyim", "öğrenciyim", "öğrencisi", "öğrenci",
    "sınıf", "sınıftayım", "okuyorum",
    # Corpus geneli — tüm dokümanlarda geçer, ayırt edici değil
    "bilgisayar", "mühendisliği", "mühendislik",
    "bölümü", "bölümde", "bölümüne", "bölümünde", "bölüm",
    "ktün", "ktun",
    # Bağlaçlar / edatlar
    "için", "ile", "ve", "veya", "ama", "de", "da",
    "mi", "mı", "mu", "mü", "bir", "bu", "şu",
    "olan", "olarak", "gibi", "var",
}

def preprocess_bm25_query(query: str) -> list[str]:
    """Stop word'leri ve noktalama işaretlerini çıkarır; içerik kelimeleri döndürür."""
    tokens = query.lower().split()
    cleaned = [t.strip("?.!,;:\"'()") for t in tokens]
    filtered = [t for t in cleaned if t and t not in _STOP_WORDS]
    return filtered if filtered else [t.strip("?.!,;:\"'()") for t in tokens]


# ── HİBRİT ARAMA ────────────────────────────────────────────────────────
def hybrid_search(
    query: str,
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    corpus_texts: list,
    corpus_answers: list,
    embedding_model: SentenceTransformer
) -> dict:
    """
    BM25 + FAISS hibrit araması yapar.
    Güven skoruna göre sonuç döndürür.
    """

    # ── BM25 skoru ──────────────────────────────────────────────────────
    tokenized_query = preprocess_bm25_query(query)
    bm25_scores     = bm25.get_scores(tokenized_query)

    # Normalize et (0-1 arasına çek)
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    # ── Semantik skor ───────────────────────────────────────────────────
    # Dolgu kelimelerden arındırılmış sorgu — FAISS için de daha temiz embedding üretir
    clean_query = " ".join(tokenized_query)
    query_vec = embedding_model.encode([clean_query])
    query_vec = query_vec / np.linalg.norm(query_vec)
    distances, indices = faiss_index.search(
        query_vec.astype(np.float32), TOP_K
    )
    semantic_scores = np.zeros(len(corpus_texts))
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(corpus_texts):
            # Negatif cosine similarity skorlarını sıfırla
            semantic_scores[idx] = max(0.0, float(score))

    # Normalize et (0-1 arasına çek) — BM25 normalizasyonuyla tutarlı olsun
    sem_max = semantic_scores.max()
    if sem_max > 0:
        semantic_scores = semantic_scores / sem_max

    # ── Hibrit skor ─────────────────────────────────────────────────────
    combined = BM25_WEIGHT * bm25_scores + SEMANTIC_WEIGHT * semantic_scores

    # Top-K sonuç
    top_indices = np.argsort(combined)[::-1][:TOP_K]
    top_score   = float(combined[top_indices[0]])

    # En iyi sonuçları topla
    top_results = []
    for i in top_indices:
        top_results.append({
            "text":   corpus_texts[i],
            "answer": corpus_answers[i],
            "score":  float(combined[i])
        })

    return {
        "top_score":   top_score,
        "top_answer":  corpus_answers[top_indices[0]],
        "top_results": top_results
    }