# Adaptive RAG System Implementation Roadmap

## Phase 1: Environment & Baseline Setup
- [x] Install dependencies: `airllm`, `faiss-cpu`, `sentence-transformers`, `rank_bm25`, `ragas`.
- [x] Initialize `Qwen/Qwen2.5-32B-Instruct` via AirLLM.
- [x] Implement a basic inference wrapper for Qwen.

## Phase 2: Indexing & Retrieval Policy
- [x] **Preprocessing:** Parse `CategorizedGeneralQuestions.json` for indexing (Instruction-Output pairs).
- [x] **Hybrid Retriever:**
    - [x] Implement BM25 (Sparse).
    - [x] Implement FAISS with Multilingual Embeddings (Dense).
    - [x] Create a weighted reciprocal rank fusion (RRF) for Hybrid results.
- [x] **Reranker:** Integrate a Cross-Encoder for top-k refinement (Running on CPU).

## Phase 3: Adaptive Decision Layer
- [x] **Query Understanding:**
    - [x] Intent detection (Course info, general, out-of-scope).
    - [x] Complexity classification (Simple vs. Multi-hop).
- [x] **Router Implementation:**
    - [x] Logic for DIRECT_LLM vs. SINGLE_RAG vs. MULTI_HOP.
    - [x] Implementation of `analyze_query` and `route_query` functions.

## Phase 4: Execution & Verification Loop
- [x] **Evidence Sufficiency Check:** Logic to determine if retrieved docs answer the query.
- [x] **Critique & Self-Reflection:**
    - [x] Faithfulness check (Does the answer contradict context?).
    - [x] Support check (Is every claim cited?).
- [x] **Iteration Logic:** Implement query rewriting and re-retrieval for low-confidence results.

## Phase 5: Advanced Flows
- [ ] **Multi-hop Logic:** Query decomposition and evidence merging. (MVP Logic in place, needs refinement)
- [x] **Abstain/Fallback Logic:** Handle "No answer" (Alternative implemented).

## Phase 6: Evaluation (RAGAS)
- [ ] Integrate RAGAS review system.
- [ ] Run evaluation against `test_set.json`.
- [ ] Calculate Faithfulness, Answer Relevancy, and Context Recall.

## Phase 7: Optimization & Finalization
- [ ] Optimize AirLLM inference (quantization, prefetching).
- [ ] Final UI/CLI interface for the Adaptive RAG system.
