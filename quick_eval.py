import sys
import json
import time
import os

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rag.embedder import BGE_M3_Embedder
from agents.graph import build_graph
from rag.retriever import (
    load_data, build_corpus, build_bm25, build_faiss, hybrid_search
)
from config import EMBEDDING_MODEL, DATA_PATH


def load_rag():
    print("Veri seti yükleniyor...")
    data = load_data(DATA_PATH)
    corpus_texts, corpus_answers = build_corpus(data)
    bm25 = build_bm25(corpus_texts)
    print("Embedding modeli yükleniyor (bge-m3)...")
    embedding_model = BGE_M3_Embedder(EMBEDDING_MODEL)
    print("FAISS indeksi oluşturuluyor...")
    faiss_index = build_faiss(corpus_texts, embedding_model)
    print("RAG hazır.\n")
    return bm25, faiss_index, corpus_texts, corpus_answers, embedding_model


def run_eval():
    if not os.path.exists("gold_set.json"):
        print("HATA: gold_set.json bulunamadı.")
        sys.exit(1)

    with open("gold_set.json", encoding="utf-8") as f:
        gold = json.load(f)

    bm25, faiss_index, corpus_texts, corpus_answers, embedding_model = load_rag()
    graph = build_graph()

    search_kwargs = dict(
        bm25=bm25,
        faiss_index=faiss_index,
        corpus_texts=corpus_texts,
        corpus_answers=corpus_answers,
        embedding_model=embedding_model,
    )

    per_question = []
    latencies = []

    print(f"{'='*60}")
    print(f"{len(gold)} soru değerlendiriliyor...")
    print(f"{'='*60}\n")

    for i, item in enumerate(gold):
        qid      = item.get("id", f"Q{i+1:02d}")
        question = item["question"]
        keywords = item.get("expected_keywords", [])

        print(f"[{qid}] {question}")

        state = {
            "user_input":        question,
            "intent":            "",
            "retrieved_context": "",
            "confidence_score":  0.0,
            "final_answer":      "",
            "error":             "",
            "bm25":              bm25,
            "faiss_index":       faiss_index,
            "corpus_texts":      corpus_texts,
            "corpus_answers":    corpus_answers,
            "embedding_model":   embedding_model,
            "retrieval_query":   "",
        }

        t0     = time.perf_counter()
        result = graph.invoke(state)
        t1     = time.perf_counter()

        latency = round(t1 - t0, 3)
        latencies.append(latency)

        answer           = result.get("final_answer", "")
        retrieval_query  = result.get("retrieval_query") or question

        # Top-5 chunk'ları grafikten bağımsız olarak al
        top5_raw      = hybrid_search(query=retrieval_query, **search_kwargs)["top_results"][:5]
        top5_previews = [r["answer"][:200] for r in top5_raw]

        answer_lower = answer.lower()
        chunks_text  = " ".join(top5_previews).lower()
        kw_in_answer = [kw for kw in keywords if kw.lower() in answer_lower]
        kw_in_top5   = [kw for kw in keywords if kw.lower() in chunks_text]

        kw_hit = f"{len(kw_in_answer)}/{len(keywords)}" if keywords else "—"
        print(f"  Latency: {latency}s  |  Keyword hit (answer): {kw_hit}")
        print(f"  Cevap: {answer[:120].replace(chr(10), ' ')}...\n")

        per_question.append({
            "id":                      qid,
            "question":                question,
            "answer":                  answer,
            "latency_sec":             latency,
            "top5_chunks_preview":     top5_previews,
            "expected_keywords":       keywords,
            "keywords_found_in_answer": kw_in_answer,
            "keywords_found_in_top5":  kw_in_top5,
        })

        if i < len(gold) - 1:
            time.sleep(0.3)

    # ── Özet metrikler ────────────────────────────────────────────────────
    n   = len(latencies)
    avg = round(sum(latencies) / n, 3)
    p95 = round(sorted(latencies)[int(n * 0.95) - 1], 3)
    mn  = round(min(latencies), 3)
    mx  = round(max(latencies), 3)

    summary = {
        "n_questions":      n,
        "avg_latency_sec":  avg,
        "p95_latency_sec":  p95,
        "min_latency_sec":  mn,
        "max_latency_sec":  mx,
    }

    output = {"summary": summary, "per_question": per_question}

    with open("quick_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Markdown rapor ────────────────────────────────────────────────────
    md = "# RAG Hızlı Değerlendirme Raporu\n\n"

    md += "## Özet Metrikler\n\n"
    md += "| Metrik | Değer |\n|---|---|\n"
    md += f"| Soru Sayısı | {n} |\n"
    md += f"| Ort. Latency | {avg}s |\n"
    md += f"| P95 Latency | {p95}s |\n"
    md += f"| Min Latency | {mn}s |\n"
    md += f"| Max Latency | {mx}s |\n"

    md += "\n## Sorular ve Cevaplar\n\n"
    md += "| ID | Soru | Cevap | Latency (s) | Keyword Hit | Doğru mu? |\n"
    md += "|---|---|---|---|---|---|\n"
    for r in per_question:
        q   = r["question"][:55].replace("|", "/")
        a   = r["answer"][:90].replace("|", "/").replace("\n", " ")
        kw  = f"{len(r['keywords_found_in_answer'])}/{len(r['expected_keywords'])}" if r["expected_keywords"] else "—"
        md += f"| {r['id']} | {q} | {a}… | {r['latency_sec']} | {kw} |  |\n"

    md += "\n## Chunk Önizlemeleri\n\n"
    for r in per_question:
        md += f"### {r['id']} — {r['question']}\n"
        for j, chunk in enumerate(r["top5_chunks_preview"], 1):
            md += f"**Chunk {j}:** {chunk[:200]}\n\n"
        md += "---\n\n"

    with open("quick_results.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("=" * 60)
    print(f"Tamamlandı.")
    print(f"  Ort. latency : {avg}s")
    print(f"  P95 latency  : {p95}s")
    print(f"  Max latency  : {mx}s")
    print("quick_results.json ve quick_results.md hazır.")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()
