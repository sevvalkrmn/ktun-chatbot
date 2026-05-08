import json
import sys
import os

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# config.py proje kökündedir; bu dosya architectures/hybrid_rag/ altında
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
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
    Her kayıt için zenginleştirilmiş index metni oluşturur.

    index_text = canonical_question + question_variants + keywords
    Bu sayede kullanıcı kısaltma ("mat kaç akts?") yazdığında
    question_variants içindeki varyantlarla eşleşir.

    corpus_texts  → BM25 ve FAISS için index metni
    corpus_answers → döndürülecek cevap (answer_full)
    """
    corpus_texts   = []
    corpus_answers = []
    seen_ids       = set()

    for item in data:
        id_   = item.get("id", "")
        soru  = item.get("canonical_question", "")
        cevap = item.get("answer_full", "")
        if not soru or not cevap:
            continue
        # Duplicate ID kontrolü
        if id_ and id_ in seen_ids:
            continue
        if id_:
            seen_ids.add(id_)

        variants = item.get("question_variants", [])
        keywords = item.get("keywords", [])

        # Zenginleştirilmiş index metni
        parts = [soru]
        if variants:
            parts.append(" ".join(variants))
        if keywords:
            parts.append(" ".join(keywords))
        index_text = " ".join(parts)

        corpus_texts.append(index_text)
        corpus_answers.append(cevap)

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

# ── HİBRİT ARAMA ────────────────────────────────────────────────────────
def hybrid_search(
    query: str,
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    corpus_texts: list,
    corpus_answers: list,
    embedding_model: SentenceTransformer
) -> dict:
    """BM25 + FAISS hibrit araması yapar."""

    # ── BM25 skoru ──────────────────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)

    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max
        # Düşük BM25 skorlarını bastır
        bm25_scores = np.where(bm25_scores < 0.1, 0.0, bm25_scores)

    # ── Semantik skor ───────────────────────────────────────────────────
    query_vec = embedding_model.encode([query])
    query_vec = query_vec / np.linalg.norm(query_vec)
    distances, indices = faiss_index.search(
        query_vec.astype(np.float32), TOP_K
    )
    semantic_scores = np.zeros(len(corpus_texts))
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(corpus_texts):
            semantic_scores[idx] = max(0.0, float(score))

    # Normalize etme — ham cosine skorlarını kullan
    # Cosine benzerliği zaten 0-1 arasında
    semantic_scores_raw = np.zeros(len(corpus_texts))
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(corpus_texts):
            semantic_scores_raw[idx] = max(0.0, float(score))
    semantic_scores = semantic_scores_raw

    # ── Hibrit skor ─────────────────────────────────────────────────────
    combined    = BM25_WEIGHT * bm25_scores + SEMANTIC_WEIGHT * semantic_scores
    top_indices = np.argsort(combined)[::-1][:TOP_K]
    top_score   = float(combined[top_indices[0]])

    top_results = [
        {
            "text":   corpus_texts[i],
            "answer": corpus_answers[i],
            "score":  float(combined[i])
        }
        for i in top_indices
    ]

    return {
        "top_score":   top_score,
        "top_answer":  corpus_answers[top_indices[0]],
        "top_results": top_results
    }


# ── HİBRİT RETRİEVER SINIFI ─────────────────────────────────────────────
class HybridRetriever:
    """
    Hybrid RAG retriever — BM25 + FAISS.
    Kullanım:
        r = HybridRetriever()
        r.build()
        result = r.search("staj kaç gün")
    """

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path      = data_path
        self.bm25           = None
        self.faiss_index    = None
        self.corpus_texts   = None
        self.corpus_answers = None
        self.embedding_model = None

    def build(self):
        """Veriyi yükler, BM25 ve FAISS indekslerini kurar."""
        data = load_data(self.data_path)
        self.corpus_texts, self.corpus_answers = build_corpus(data)
        self.bm25            = build_bm25(self.corpus_texts)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.faiss_index     = build_faiss(self.corpus_texts, self.embedding_model)
        print(f"Index hazır: {len(self.corpus_texts)} doküman")

    def search(self, query: str) -> dict:
        """Hibrit arama yapar, en iyi sonucu döndürür."""
        return hybrid_search(
            query           = query,
            bm25            = self.bm25,
            faiss_index     = self.faiss_index,
            corpus_texts    = self.corpus_texts,
            corpus_answers  = self.corpus_answers,
            embedding_model = self.embedding_model
        )
