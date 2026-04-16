from agents.state import AgentState
from rag.retriever import hybrid_search, preprocess_bm25_query
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config import MODEL_NAME, LOAD_IN_4BIT, SYSTEM_PROMPT
from config import (
    HIGH_CONFIDENCE, MID_CONFIDENCE, LOW_CONFIDENCE,
    SYSTEM_PROMPT
)

# ── 1. ÖN FİLTRE & NİYET BELIRLEME NODE'U ──────────────────────────────
def intent_node(state: AgentState) -> AgentState:
    """
    Kullanıcının ne istediğini anlar.
    LLM kullanmaz — kural tabanlı çalışır, hızlı ve güvenilir.
    """
    user_input = state["user_input"].lower().strip()

    # Küfür/hakaret filtresi — önce kontrol et
    kufurler = [
        "küfür", "sövme", "orospu", "amk", "bok", "sik", "göt",
        "salak", "aptal", "gerizekalı", "mal", "oç", "piç"
    ]
    if any(k in user_input for k in kufurler):
        return {**state, "intent": "kufur"}

    # Konu dışı kontrolü
    konu_disi = [
        "hava", "film", "müzik", "yemek tarifi", "futbol",
        "basketbol", "dizi", "oyun", "sevgili", "aşk",
        "para kazan", "kripto", "borsa"
    ]
    if any(k in user_input for k in konu_disi):
        return {**state, "intent": "konu_disi"}

    # Selamlama kontrolü — yalnızca ≤5 kelimelik kısa mesajlarda
    selamlar = [
        "merhaba", "selam", "slm", "hey", "hi", "hello",
        "iyi günler", "iyi akşamlar", "iyi sabahlar", "mrb"
    ]
    selamlar_tam_kelime = ["sa"]
    kelimeler = user_input.split()
    is_kisa   = len(kelimeler) <= 5
    eslesen_selam = (
        any(s in user_input for s in selamlar) or
        any(s in kelimeler for s in selamlar_tam_kelime)
    )
    if is_kisa and eslesen_selam:
        return {**state, "intent": "selamlama"}

    # Vedalaşma
    vedalar = [
        "görüşürüz", "hoşça kal", "güle güle", "bye",
        "tamam görüşürüz", "görüşmek üzere", "çıkıyorum"
    ]
    if any(v in user_input for v in vedalar):
        return {**state, "intent": "veda"}

    # Kimlik sorusu
    kimlik = [
        "sen kimsin", "ne yaparsın", "kim sin", "adın ne",
        "ne iş yapıyorsun", "nasıl yardımcı"
    ]
    if any(k in user_input for k in kimlik):
        return {**state, "intent": "kimlik"}

    # Hal hatır — akademik değil ama nezaket cevabı verilmeli
    hal_hatir = [
        "nasılsın", "naber", "ne haber", "iyimisin",
        "nasıl gidiyor", "iyi misin"
    ]
    if any(h in user_input for h in hal_hatir):
        return {**state, "intent": "hal_hatir"}

    # Takvim isteği
    takvim = [
        "takvim", "ekle", "calendar", "hatırlat", "hatırlatıcı"
    ]
    if any(t in user_input for t in takvim):
        return {**state, "intent": "takvim_ekle"}

    # Anlamsız girdi — çok kısa veya sadece harf/rakam karışımı
    if len(user_input) < 3:
        return {**state, "intent": "anlamsiz"}

    # Akademik soru — stop word'lerden arındırılmış temiz sorgu kullan
    clean_tokens = preprocess_bm25_query(state["user_input"])
    retrieval_query = " ".join(clean_tokens)
    return {**state, "intent": "bilgi_sorusu", "retrieval_query": retrieval_query}

# ── 2. RAG NODE ─────────────────────────────────────────────────────────
def rag_node(state: AgentState) -> AgentState:
    """
    Kullanıcının sorusunu veri setinde arar.
    Güven skoruna göre ne yapılacağına karar verir.
    """
    # Retriever nesneleri state'ten gelecek (graph.py'da yüklenecek)
    bm25            = state.get("bm25")
    faiss_index     = state.get("faiss_index")
    corpus_texts    = state.get("corpus_texts")
    corpus_answers  = state.get("corpus_answers")
    embedding_model = state.get("embedding_model")

    result = hybrid_search(
        query           = state.get("retrieval_query", state["user_input"]),
        bm25            = bm25,
        faiss_index     = faiss_index,
        corpus_texts    = corpus_texts,
        corpus_answers  = corpus_answers,
        embedding_model = embedding_model
    )

    top_score  = result["top_score"]
    top_answer = result["top_answer"]
    top_results = result["top_results"]

    # Sorgu "yandal" içermiyorsa yandal belgelerini filtrele
    retrieval_query = state.get("retrieval_query", state["user_input"])
    if "yandal" not in retrieval_query.lower():
        filtered = [r for r in top_results if "yandal" not in r.get("text", "").lower()]
        if filtered:
            top_results = filtered
            top_answer  = filtered[0]["answer"]
            top_score   = filtered[0]["score"]

    # Sorgu kaç içerik kelimesi içeriyor?
    query_word_count = len(retrieval_query.strip().split())

    # Güven skoruna göre karar ver
    # Vague sorgu (≤2 kelime): LLM top-3 ile sentez yaparak daha kapsamlı cevap üretir
    if top_score >= HIGH_CONFIDENCE and query_word_count >= 3:
        # Spesifik sorgu + yüksek güven → direkt veri setinden cevap dön
        return {
            **state,
            "confidence_score":  top_score,
            "retrieved_context": top_answer,
            "final_answer":      top_answer,
            "intent":            "dogrudan_cevap"
        }

    elif top_score >= MID_CONFIDENCE:
        # Bağlamı hazırla, LLM sentez yapacak
        context = "\n\n".join([r["answer"] for r in top_results[:3]])
        return {
            **state,
            "confidence_score":  top_score,
            "retrieved_context": context,
            "intent":            "llm_sentez"
        }

    else:
        # Güven düşük — bilmiyorum
        return {
            **state,
            "confidence_score": top_score,
            "retrieved_context": "",
            "intent":            "dusuk_guven"
        }
    
    # ── 3. LLM SENTEZ NODE'U ─────────────────────────────────────────────────
def llm_node(state: AgentState) -> AgentState:
    """
    RAG'dan gelen bağlamı kullanarak LLM ile cevap üretir.
    Yalnızca orta güven skorlu sorgular buraya gelir.
    """
    model     = state.get("model")
    tokenizer = state.get("tokenizer")
    context   = state["retrieved_context"]
    question  = state["user_input"]

    prompt = (
        "Aşağıdaki BAĞLAM bilgisine dayanarak soruyu yanıtla.\n"
        + "BAĞLAM dışında bilgi ekleme.\n\n"
        + "BAĞLAM:\n"
        + context
        + "\n\nSORU: " + question
        + "\nCEVAP:"
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,    # Düşük — tutarlı, yaratıcı değil
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Sadece üretilen kısmı al, prompt'u çıkar
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer    = tokenizer.decode(generated, skip_special_tokens=True)

    return {**state, "final_answer": answer.strip()}


# ── 4. SELAMLAMA NODE'U ──────────────────────────────────────────────────
def selamlama_node(state: AgentState) -> AgentState:
    """LLM kullanmaz — sabit karşılama mesajı döner."""
    return {
        **state,
        "final_answer": (
            "Merhaba! Ben KTÜN Bilgisayar Mühendisliği Bölümü "
            "Akademik Bilgi Asistanıyım. Dersler, staj, yatay geçiş, "
            "Erasmus ve diğer akademik konularda sana yardımcı olabilirim."
        )
    }


# ── 5. KİMLİK NODE'U ─────────────────────────────────────────────────────
def kimlik_node(state: AgentState) -> AgentState:
    """Sabit kimlik cevabı döner."""
    return {
        **state,
        "final_answer": (
            "Ben KTÜN Bilgisayar Mühendisliği Bölümü Akademik Bilgi "
            "Asistanıyım. Bölümle ilgili sorularını yanıtlamak için "
            "buradayım. Sana nasıl yardımcı olabilirim?"
        )
    }


# ── 6. DÜŞÜK GÜVEN NODE'U ────────────────────────────────────────────────
def dusuk_guven_node(state: AgentState) -> AgentState:
    """Veri setinde bulunamayan sorular için yönlendirme yapar."""
    return {
        **state,
        "final_answer": (
            "Bu konuda kesin bir bilgim bulunmuyor. "
            "Daha doğru bilgi için bölüm sekreterliğine "
            "veya danışman hocanıza başvurmanızı öneririm."
        )
    }


# ── 7. KONU DIŞI NODE'U ──────────────────────────────────────────────────
def konu_disi_node(state: AgentState) -> AgentState:
    """Akademik dışı sorular için sabit cevap döner."""
    return {
        **state,
        "final_answer": (
            "Üzgünüm, yalnızca KTÜN Bilgisayar Mühendisliği Bölümü "
            "ile ilgili akademik konularda yardımcı olabiliyorum."
        )
    }

# ── KÜFÜR NODE'U ─────────────────────────────────────────────────────────
def kufur_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": (
            "Bu tür bir isteğe yardımcı olamam. "
            "Akademik konularda sormak istediğin bir şey var mı?"
        )
    }

# ── VEDA NODE'U ──────────────────────────────────────────────────────────
def veda_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": "Görüşmek üzere! Başarılar dilerim."
    }

# ── HAL HATIR NODE'U ─────────────────────────────────────────────────────
def hal_hatir_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": (
            "İyiyim, teşekkür ederim! "
            "Akademik konularda sana nasıl yardımcı olabilirim?"
        )
    }

# ── ANLAMSIZ GİRDİ NODE'U ────────────────────────────────────────────────
def anlamsiz_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": (
            "Sorunuzu anlayamadım. "
            "Dersler, staj, yatay geçiş gibi konularda "
            "sormak istediğiniz bir şey var mı?"
        )
    }