import re
from openai import OpenAI
from agents.state import AgentState
from rag.retriever import hybrid_search, preprocess_bm25_query
from tools.yemekhane_tool import get_day_menu
from config import SYSTEM_PROMPT, HIGH_CONFIDENCE, MID_CONFIDENCE, OPENAI_API_KEY, OPENAI_MODEL

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


# ── Yemekhane menü sorgusu tespiti ──────────────────────────────────────
_MENU_SIGNALS = [
    "menü", "menu", "yemek listesi", "yemekte ne", "ne yemek",
    "bugün ne var", "bugünkü yemek", "öğle yemeğ", "akşam yemeğ",
    "bugün yemek", "yemekte ne var", "ne pişmiş", "ne çıkmış",
]

def _is_menu_query(text: str) -> bool:
    """Yemekhane MENÜ sorusu mu? (rezervasyon/ücret/konum soruları RAG'a gider)"""
    t = text.lower()
    if any(s in t for s in _MENU_SIGNALS):
        return True
    if "yemek" in t and any(s in t for s in
                            ["bugün", "yarın", "liste", "ne var", "var mı",
                             "çıkıyor", "akşam", "öğle"]):
        return True
    return False


# ── E-posta (mail) niyeti tespiti ───────────────────────────────────────
def _is_mail_query(text: str) -> bool:
    """Kullanıcı bir mail yazılmasını/gönderilmesini mi istiyor?"""
    t = text.lower()
    has_mail_word = any(w in t for w in ["mail", "e-posta", "eposta", "e posta"])
    if not has_mail_word:
        return False
    # 'mail' geçse de bir eylem (yaz/gönder/at/hazırla/taslak) olmalı
    return any(v in t for v in
               ["yaz", "gönder", "at ", "atar", "hazırla", "oluştur", "taslak", "çek"])


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

    if _is_mail_query(user_input):
        return {**state, "intent": "mail"}

    if _is_menu_query(user_input):
        return {**state, "intent": "yemekhane"}

    expanded        = _expand_sinif_donem(state["user_input"])
    clean_tokens    = preprocess_bm25_query(expanded)
    retrieval_query = " ".join(clean_tokens)
    return {**state, "intent": "bilgi_sorusu", "retrieval_query": retrieval_query}


# ── 2. RAG NODE ──────────────────────────────────────────────────────────
def rag_node(state: AgentState) -> AgentState:
    # RAG kaynakları: state'te varsa onları kullan (CLI/Flask yolu), yoksa
    # singleton'dan al (FastAPI/agent yolu — checkpointer state'i şişirmesin diye)
    bm25 = state.get("bm25")
    if bm25 is None:
        from rag.resources import get_resources
        r = get_resources()
        bm25, faiss_index = r["bm25"], r["faiss_index"]
        corpus_texts, corpus_answers = r["corpus_texts"], r["corpus_answers"]
        embedding_model = r["embedding_model"]
    else:
        faiss_index     = state.get("faiss_index")
        corpus_texts    = state.get("corpus_texts")
        corpus_answers  = state.get("corpus_answers")
        embedding_model = state.get("embedding_model")

    query = state.get("retrieval_query") or state["user_input"]
    result = hybrid_search(
        query           = query,
        bm25            = bm25,
        faiss_index     = faiss_index,
        corpus_texts    = corpus_texts,
        corpus_answers  = corpus_answers,
        embedding_model = embedding_model,
    )

    top_score   = result["top_score"]
    top_results = result["top_results"]
    retrieval_query = query.lower()

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


# ── 3. LLM SENTEZ NODE (OpenAI — gpt-4o-mini) ───────────────────────────
def llm_node(state: AgentState) -> AgentState:
    context  = state["retrieved_context"]
    question = state["user_input"]

    client   = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"BAĞLAM:\n{context}\n\nSORU: {question}\nCEVAP:"},
        ],
        temperature=0,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    return {**state, "final_answer": answer}


# ── YEMEKHANE NODE (canlı veri — KTÜN sitesi + vision) ──────────────────
def yemekhane_node(state: AgentState) -> AgentState:
    try:
        d = get_day_menu()
    except Exception as e:
        print(f"Yemekhane aracı hatası: {e}")
        return {**state, "final_answer": (
            "Yemek listesine şu an ulaşılamıyor. Güncel listeyi KTÜN duyurularından "
            "kontrol edebilirsiniz: https://www.ktun.edu.tr/tr/Universite/TumDuyurular"
        )}

    if d["menu"]:
        answer = (
            f"Bugün ({d['tarih']} {d['gun_adi']}) KTÜN yemekhane menüsü:\n"
            f"{d['menu']}"
        )
    else:
        answer = (
            f"Bugün ({d['tarih']} {d['gun_adi']}) için yemekhane menüsünde yemek "
            f"görünmüyor; hafta sonu veya tatil olabilir."
        )
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
