"""
RAG kaynakları için süreç-içi singleton.

Checkpointer (LangGraph durum kalıcılığı) state'i serialize ettiğinden, ağır ve
serialize edilemeyen nesneleri (FAISS indeksi, bge-m3 modeli) graph state'inde
TAŞIMAYIZ. Bunun yerine node'lar buradan, bir kez yüklenmiş halini alır.
"""
import os

from rag.retriever import load_data, build_corpus, build_bm25, build_faiss
from rag.embedder import BGE_M3_Embedder
from config import DATA_PATH, EMBEDDING_MODEL

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCES = None


def get_resources() -> dict:
    """bge-m3 + FAISS + BM25 + corpus'u bir kez kurar, sonra cache'ten döner."""
    global _RESOURCES
    if _RESOURCES is None:
        print("RAG kaynakları yükleniyor (singleton)...")
        data = load_data(os.path.join(_ROOT, DATA_PATH))
        corpus_texts, corpus_answers = build_corpus(data)
        embedding_model = BGE_M3_Embedder(EMBEDDING_MODEL)
        _RESOURCES = {
            "bm25": build_bm25(corpus_texts),
            "faiss_index": build_faiss(corpus_texts, embedding_model),
            "corpus_texts": corpus_texts,
            "corpus_answers": corpus_answers,
            "embedding_model": embedding_model,
        }
        print("RAG kaynakları hazır.")
    return _RESOURCES
