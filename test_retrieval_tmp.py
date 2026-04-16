import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import load_data, build_corpus, build_bm25, build_faiss, hybrid_search
from sentence_transformers import SentenceTransformer
from config import DATA_PATH, EMBEDDING_MODEL, HIGH_CONFIDENCE, MID_CONFIDENCE

print("Veri yükleniyor...")
data = load_data(DATA_PATH)
corpus_texts, corpus_answers = build_corpus(data)
bm25 = build_bm25(corpus_texts)
emb  = SentenceTransformer(EMBEDDING_MODEL)
faiss_index = build_faiss(corpus_texts, emb)
print("Hazır.\n")

queries = [
    "staj hakkında bilgi ver",
    "staj hakkında bilgi verir misin",
    "staj nedir",
    "staj kaç gün",
    "staj kaç gün ne zaman yapılır bunlar hakkında bilgi ver",
    "Ben bilgisayar mühendisliği 2. sınıf öğrencisiyim. Bana staj hakkında bilgi ver",
]

for q in queries:
    r = hybrid_search(q, bm25, faiss_index, corpus_texts, corpus_answers, emb)
    score = r["top_score"]
    seviye = "HIGH" if score >= HIGH_CONFIDENCE else ("MID" if score >= MID_CONFIDENCE else "LOW")
    print(f"Sorgu : {q!r}")
    print(f"Skor  : {score:.3f} [{seviye}]")
    print(f"Cevap : {r['top_answer'][:120].strip()}...")
    print()
