import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from config import (
    MODEL_NAME, LOAD_IN_4BIT, DEVICE, SYSTEM_PROMPT, TOP_K,
    DATA_PATH, EMBEDDING_MODEL,
)

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_set.json")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")

# Embedding modeli (metrikler için)
_EVAL_EMBED_MODEL = None

def get_eval_embed_model():
    global _EVAL_EMBED_MODEL
    if _EVAL_EMBED_MODEL is None:
        print("Değerlendirme embedding modeli yükleniyor...")
        _EVAL_EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding modeli hazır.")
    return _EVAL_EMBED_MODEL


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ── Embedding tabanlı RAGAS benzeri metrikler ─────────────────────────────────

def compute_answer_relevancy(question: str, answer: str, emb: SentenceTransformer) -> float:
    """cosine(soru_emb, cevap_emb)"""
    q_emb = emb.encode(question)
    a_emb = emb.encode(answer)
    return max(0.0, cosine(q_emb, a_emb))


def compute_faithfulness(answer: str, context: str, emb: SentenceTransformer) -> float:
    """
    Cevabı cümlelere böl, her cümle için context ile max cosine al.
    Ortalama = faithfulness skoru.
    """
    sentences = [s.strip() for s in answer.replace(".", ".\n").split("\n") if s.strip()]
    if not sentences:
        return 0.0
    ctx_emb = emb.encode(context)
    scores = []
    for sent in sentences:
        s_emb = emb.encode(sent)
        scores.append(max(0.0, cosine(s_emb, ctx_emb)))
    return float(np.mean(scores))


def compute_context_precision(context: str, ground_truth: str, emb: SentenceTransformer) -> float:
    """cosine(context_emb, ground_truth_emb) — context ne kadar doğru bilgi içeriyor"""
    c_emb = emb.encode(context)
    g_emb = emb.encode(ground_truth)
    return max(0.0, cosine(c_emb, g_emb))


def compute_context_recall(context: str, ground_truth: str, emb: SentenceTransformer) -> float:
    """
    ground_truth cümlelerinin context'te ne kadarının karşılığı var.
    Her gt cümlesi için context ile max cosine → ortalama.
    """
    sentences = [s.strip() for s in ground_truth.replace(".", ".\n").split("\n") if s.strip()]
    if not sentences:
        return 0.0
    ctx_emb = emb.encode(context)
    scores = []
    for sent in sentences:
        s_emb = emb.encode(sent)
        scores.append(max(0.0, cosine(s_emb, ctx_emb)))
    return float(np.mean(scores))


# ── Hybrid RAG sarmalayıcı ────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self):
        from architectures.hybrid_rag.retriever import (
            load_data, build_corpus, build_bm25, build_faiss, hybrid_search
        )
        print("Hybrid RAG bileşenleri yükleniyor...")
        data = load_data(DATA_PATH)
        corpus_texts, corpus_answers = build_corpus(data)
        self._bm25            = build_bm25(corpus_texts)
        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._faiss_index     = build_faiss(corpus_texts, self._embedding_model)
        self._corpus_texts    = corpus_texts
        self._corpus_answers  = corpus_answers
        self._hybrid_search   = hybrid_search
        print("Hybrid RAG hazır.")

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        result = self._hybrid_search(
            query           = query,
            bm25            = self._bm25,
            faiss_index     = self._faiss_index,
            corpus_texts    = self._corpus_texts,
            corpus_answers  = self._corpus_answers,
            embedding_model = self._embedding_model,
        )
        return [{"content": r["answer"]} for r in result["top_results"][:k]]


# ── Retriever yükleyici ───────────────────────────────────────────────────────
def load_retriever(architecture: str):
    if architecture == "hybrid":
        return HybridRetriever()

    elif architecture == "cograg":
        from architectures.cog_rag.retriever import CogRAGRetriever
        retriever = CogRAGRetriever()
        retriever.build()
        return retriever

    elif architecture == "colbert":
        from architectures.colbert_rag.retriever import ColBERTRetriever
        return ColBERTRetriever()

    else:
        raise ValueError(f"Bilinmeyen mimari: {architecture}")


# ── LLM yükleme ──────────────────────────────────────────────────────────────
def load_model():
    print(f"Model yükleniyor: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit           = LOAD_IN_4BIT,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",
    )
    print("Model yüklendi.")
    return model, tokenizer


def generate_answer(model, tokenizer, context: str, question: str) -> str:
    prompt = (
        "Asagidaki BAGLAN bilgisine dayanarak soruyu yanitla.\n"
        "BAGLAN disinda bilgi ekleme. Sadece Turkce cevap ver.\n\n"
        "BAGLAN:\n" + context + "\n\nSORU: " + question + "\nCEVAP:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens   = 256,
            temperature      = 0.1,
            do_sample        = True,
            pad_token_id     = tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── CogRAG için search sarmalayıcı ────────────────────────────────────────────
def _cograg_search(retriever, query: str, k: int = TOP_K) -> list[dict]:
    result = retriever.search(query)
    return [{"content": r["answer"]} for r in result.get("top_results", [])[:k]]


# ── Ana değerlendirme fonksiyonu ──────────────────────────────────────────────
def evaluate_architecture(retriever, model, tokenizer,
                           test_data: list, architecture: str) -> dict:
    emb   = get_eval_embed_model()
    rows  = []
    total = len(test_data)

    metrics_acc = {
        "faithfulness":      [],
        "answer_relevancy":  [],
        "context_precision": [],
        "context_recall":    [],
    }

    for i, record in enumerate(test_data, 1):
        question     = record["instruction"]
        ground_truth = record.get("ground_truth", record.get("output", ""))

        print(f"  [{i}/{total}] {question[:60]}")

        # Retrieval
        if architecture == "cograg":
            chunks = _cograg_search(retriever, question, k=TOP_K)
        else:
            chunks = retriever.search(question, k=TOP_K)

        retrieved_context = "\n".join(
            c["content"] for c in chunks if c.get("content")
        )

        # LLM ile cevap üret
        answer = generate_answer(model, tokenizer, retrieved_context, question)

        # Embedding tabanlı metrikler
        ar  = compute_answer_relevancy(question, answer, emb)
        fa  = compute_faithfulness(answer, retrieved_context, emb)
        cp  = compute_context_precision(retrieved_context, ground_truth, emb)
        cr  = compute_context_recall(retrieved_context, ground_truth, emb)

        metrics_acc["answer_relevancy"].append(ar)
        metrics_acc["faithfulness"].append(fa)
        metrics_acc["context_precision"].append(cp)
        metrics_acc["context_recall"].append(cr)

        rows.append({
            "question":          question,
            "answer":            answer,
            "ground_truth":      ground_truth,
            "context_snippet":   retrieved_context[:200],
            "answer_relevancy":  round(ar, 4),
            "faithfulness":      round(fa, 4),
            "context_precision": round(cp, 4),
            "context_recall":    round(cr, 4),
        })

        print(f"    AR={ar:.3f}  FA={fa:.3f}  CP={cp:.3f}  CR={cr:.3f}")

    scores = {k: float(np.mean(v)) for k, v in metrics_acc.items()}
    scores["rows"] = rows
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Mimarisi Değerlendirme")
    parser.add_argument(
        "--architecture",
        choices=["hybrid", "cograg", "colbert"],
        required=True,
        help="Değerlendirilecek mimari",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Test edilecek kayıt sayısı (varsayılan: 5)",
    )
    args = parser.parse_args()
    arch = args.architecture

    # Test setini yükle
    print(f"Test seti yükleniyor: {TEST_SET_PATH}")
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    test_data = raw["sorular"]
    print(f"  {len(test_data)} test kaydı yüklendi. İlk {args.n} kullanılacak.")

    # Bileşenleri yükle
    retriever        = load_retriever(arch)
    model, tokenizer = load_model()

    # Değerlendir
    print(f"\n{arch.upper()} değerlendiriliyor...")
    scores = evaluate_architecture(retriever, model, tokenizer, test_data[:args.n], arch)

    # Sonuçları kaydet
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{arch}_results.json")
    save_scores = {k: v for k, v in scores.items() if k != "rows"}
    save_scores["detail"] = scores["rows"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_scores, f, ensure_ascii=False, indent=2)
    print(f"\nSonuçlar kaydedildi: {out_path}")

    # Özet
    print(f"\n{'='*40}")
    print(f"MİMARİ : {arch.upper()}  (n={args.n})")
    print(f"{'='*40}")
    print(f"Faithfulness       : {scores['faithfulness']:.3f}")
    print(f"Answer Relevancy   : {scores['answer_relevancy']:.3f}")
    print(f"Context Precision  : {scores['context_precision']:.3f}")
    print(f"Context Recall     : {scores['context_recall']:.3f}")


if __name__ == "__main__":
    main()
