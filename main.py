from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

from agents.graph import build_graph
from rag.retriever import load_data, build_corpus, build_bm25, build_faiss
from config import MODEL_NAME, LOAD_IN_4BIT, EMBEDDING_MODEL, DATA_PATH


def load_model():
    """Qwen2.5-7B modelini 4-bit kuantizasyonla yükler."""
    print(f"Model yükleniyor: {MODEL_NAME}")
    print("İlk çalıştırmada model internetten inecek (~4.5GB), lütfen bekle...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Model yüklendi.")
    return model, tokenizer


def load_rag():
    """RAG bileşenlerini hazırlar."""
    print("Veri seti yükleniyor...")
    data = load_data(DATA_PATH)

    print("Corpus oluşturuluyor...")
    corpus_texts, corpus_answers = build_corpus(data)

    print("BM25 indeksi oluşturuluyor...")
    bm25 = build_bm25(corpus_texts)

    print("Embedding modeli yükleniyor...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("FAISS indeksi oluşturuluyor...")
    faiss_index = build_faiss(corpus_texts, embedding_model)

    print("RAG hazır.")
    return bm25, faiss_index, corpus_texts, corpus_answers, embedding_model


def main():
    # Bileşenleri yükle
    model, tokenizer             = load_model()
    bm25, faiss_index, corpus_texts, corpus_answers, embedding_model = load_rag()

    # Graph'ı kur
    graph = build_graph()

    print("\n" + "="*50)
    print("KTÜN Akademik Bilgi Asistanı hazır.")
    print("Çıkmak için 'q' yaz.")
    print("="*50 + "\n")

    while True:
        user_input = input("Sen: ").strip()

        if user_input.lower() == "q":
            print("Görüşmek üzere!")
            break

        if not user_input:
            continue

        state = {
            "user_input":        user_input,
            "intent":            "",
            "retrieved_context": "",
            "confidence_score":  0.0,
            "final_answer":      "",
            "error":             "",
            "model":             model,
            "tokenizer":         tokenizer,
            "bm25":              bm25,
            "faiss_index":       faiss_index,
            "corpus_texts":      corpus_texts,
            "corpus_answers":    corpus_answers,
            "embedding_model":   embedding_model,
            "retrieval_query":   "",
        }

        result = graph.invoke(state)

        answer = result.get("final_answer", "Bir hata oluştu.")
        print(f"\nAsistan: {answer}\n")


if __name__ == "__main__":
    main()
