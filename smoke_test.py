import sys
sys.stdout.reconfigure(encoding="utf-8")

from rag.embedder import BGE_M3_Embedder
from agents.graph import build_graph
from rag.retriever import load_data, build_corpus, build_bm25, build_faiss
from config import EMBEDDING_MODEL, DATA_PATH

data = load_data(DATA_PATH)
corpus_texts, corpus_answers = build_corpus(data)
bm25 = build_bm25(corpus_texts)
embedding_model = BGE_M3_Embedder(EMBEDDING_MODEL)
faiss_index = build_faiss(corpus_texts, embedding_model)
graph = build_graph()

sorular = [
    "Veri biliminin temelleri dersini hangi hoca veriyor?",
    "Staja kabul aldıktan kaç gün sonra başlayabilirim?",
    "Staj başvuru sonuçları ayın 15'inde açıklandı, en erken ne zaman staja başlarım?",
    "Veri Yapıları dersine hangi hoca giriyor?",
    "Staj defterini Ağustos'ta ne zaman teslim etmeliyim?",
]

for soru in sorular:
    state = {
        "user_input": soru, "intent": "", "retrieved_context": "",
        "confidence_score": 0.0, "final_answer": "", "error": "",
        "bm25": bm25, "faiss_index": faiss_index,
        "corpus_texts": corpus_texts, "corpus_answers": corpus_answers,
        "embedding_model": embedding_model, "retrieval_query": "",
    }
    result = graph.invoke(state)
    print(f"\nSORU: {soru}")
    print(f"  skor={result.get('confidence_score'):.3f}")
    print(f"  CEVAP: {result.get('final_answer')}")

print("\n=== TEST BITTI ===")
