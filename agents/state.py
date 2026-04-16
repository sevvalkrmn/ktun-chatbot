from typing import TypedDict, Any

class AgentState(TypedDict):
    # Kullanıcının sorusu
    user_input: str

    # Genişletilmiş retrieval sorgusu (intent_node tarafından doldurulur)
    retrieval_query: str

    # Niyet
    intent: str

    # RAG'dan gelen bağlam
    retrieved_context: str

    # RAG güven skoru
    confidence_score: float

    # Final cevap
    final_answer: str

    # Hata mesajı
    error: str

    # Model nesneleri — her sorguda tekrar yüklenmemesi için
    model:           Any
    tokenizer:       Any
    bm25:            Any
    faiss_index:     Any
    corpus_texts:    Any
    corpus_answers:  Any
    embedding_model: Any
