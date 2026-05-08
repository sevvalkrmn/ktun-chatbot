import re
import ollama
from agents.state import AgentState
from rag.retriever import hybrid_search, preprocess_bm25_query
from config import SYSTEM_PROMPT, HIGH_CONFIDENCE, MID_CONFIDENCE, OLLAMA_MODEL, OLLAMA_HOST

# 1. sınıf = DÖNEM 1-2, 2. sınıf = DÖNEM 3-4, 3. sınıf = DÖNEM 5-6, 4. sınıf = DÖNEM 7-8
_SINIF_DONEM_MAP = {
    "1": "DÖNEM 1 DÖNEM 2",
    "2": "DÖNEM 3 DÖNEM 4",
    "3": "DÖNEM 5 DÖNEM 6",
    "4": "DÖNEM 7 DÖNEM 8",
}

def _expand_sinif_donem(query: str) -> str:
    """'X. sınıf' → 'DÖNEM 2X-1 DÖNEM 2X' genişletmesi yapar."""
    match = re.search(r'(\d)\.\s*s[ıi]n[ıi]f', query, re.IGNORECASE)
    if match:
        sinif = match.group(1)
        if sinif in _SINIF_DONEM_MAP:
            return re.sub(
                r'\d+\.\s*s[ıi]n[ıi]f',
                _SINIF_DONEM_MAP[sinif],
                query,
                flags=re.IGNORECASE
            )
    return query


# ── 1. NİYET BELİRLEME ──────────────────────────────────────────────────
def intent_node(state: AgentState) -> AgentState:
    user_input = state["user_input"].lower().strip()
    kelimeler  = user_input.split()
    is_kisa    = len(kelimeler) <= 5

    selamlar = ["merhaba", "selam", "slm", "hey", "hi", "hello",
                "iyi günler", "iyi akşamlar", "iyi sabahlar", "mrb"]
    if is_kisa and any(s in user_input for s in selamlar):
        return {**state, "intent": "selamlama"}

    vedalar = ["görüşürüz", "hoşça kal", "güle güle", "bye", "görüşmek üzere", "çıkıyorum"]
    if any(v in user_input for v in vedalar):
        return {**state, "intent": "veda"}

    kimlik = ["sen kimsin", "ne yaparsın", "adın ne", "nasıl yardımcı", "kim sin"]
    if any(k in user_input for k in kimlik):
        return {**state, "intent": "kimlik"}

    hal_hatir = ["nasılsın", "naber", "ne haber", "iyimisin", "nasıl gidiyor"]
    if any(h in user_input for h in hal_hatir):
        return {**state, "intent": "hal_hatir"}

    expanded        = _expand_sinif_donem(state["user_input"])
    clean_tokens    = preprocess_bm25_query(expanded)
    retrieval_query = " ".join(clean_tokens)
    return {**state, "intent": "bilgi_sorusu", "retrieval_query": retrieval_query}


# ── 2. RAG NODE ──────────────────────────────────────────────────────────
def rag_node(state: AgentState) -> AgentState:
    result = hybrid_search(
        query           = state.get("retrieval_query", state["user_input"]),
        bm25            = state.get("bm25"),
        faiss_index     = state.get("faiss_index"),
        corpus_texts    = state.get("corpus_texts"),
        corpus_answers  = state.get("corpus_answers"),
        embedding_model = state.get("embedding_model")
    )

    top_score   = result["top_score"]
    top_results = result["top_results"]
    retrieval_query = state.get("retrieval_query", state["user_input"]).lower()

    # Erasmus filtresi — sorguda "erasmus" yoksa erasmus chunk'larını dışla
    erasmus_keywords = ["erasmus", "hareketlilik", "exchange"]
    if not any(k in retrieval_query for k in erasmus_keywords):
        filtered = [r for r in top_results
                    if "erasmus" not in r.get("answer", "").lower()[:80]]
        if filtered:
            top_results = filtered
            top_score   = filtered[0]["score"]

    if top_score < MID_CONFIDENCE:
        return {**state, "confidence_score": top_score,
                "retrieved_context": "", "intent": "dusuk_guven"}

    context = "\n\n".join([r["answer"] for r in top_results[:3]])
    return {**state, "confidence_score": top_score,
            "retrieved_context": context, "intent": "llm_sentez"}


# ── 3. LLM SENTEZ NODE (Ollama — Gemma4) ────────────────────────────────
def llm_node(state: AgentState) -> AgentState:
    context  = state["retrieved_context"]
    question = state["user_input"]

    client   = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"BAĞLAM:\n{context}\n\nSORU: {question}\nCEVAP:"},
        ],
        options={"temperature": 0.1, "num_predict": 512},
    )

    answer = response.message.content.strip()
    return {**state, "final_answer": answer}


# ── DİĞER NODE'LAR ───────────────────────────────────────────────────────
def selamlama_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": (
        "Merhaba! Ben KTÜN Bilgisayar Mühendisliği Bölümü Akademik Bilgi Asistanıyım. "
        "Dersler, staj, yatay geçiş, Erasmus ve diğer akademik konularda sana yardımcı olabilirim."
    )}

def kimlik_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": (
        "Ben KTÜN Bilgisayar Mühendisliği Bölümü Akademik Bilgi Asistanıyım. "
        "Bölümle ilgili sorularını yanıtlamak için buradayım."
    )}

def veda_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": "Görüşmek üzere! Başarılar dilerim."}

def hal_hatir_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": (
        "İyiyim, teşekkür ederim! Akademik konularda sana nasıl yardımcı olabilirim?"
    )}

def dusuk_guven_node(state: AgentState) -> AgentState:
    return {**state, "final_answer": (
        "Bu konuda kesin bilgim bulunmuyor. "
        "Daha doğru bilgi için bölüm sekreterliğine veya danışman hocanıza başvurabilirsiniz."
    )}
