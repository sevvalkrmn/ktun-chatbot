"""
Birleşik agent grafiği:
  - Bizim mevcut node'larımız (intent, rag, llm, yemekhane, selamlama...)
  - + E-posta agent'ı (taslak → insan onayı → Gmail'den gönderim)

RAG ağır nesneleri state'te taşınmaz (checkpointer serialize eder); rag_node
bunları rag.resources.get_resources() singleton'ından alır.
"""
import json
from typing import TypedDict, Annotated, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from agents.nodes import (
    intent_node,
    rag_node,
    llm_node,
    selamlama_node,
    kimlik_node,
    veda_node,
    hal_hatir_node,
    dusuk_guven_node,
    yemekhane_node,
)
from config import OPENAI_API_KEY
from agent_tools import send_email_via_user_account


# ── State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:          Annotated[List, add_messages]
    user_input:        str
    intent:            str
    retrieval_query:   Optional[str]
    retrieved_context: Optional[str]
    confidence_score:  Optional[float]
    final_answer:      Optional[str]
    draft_email:       Optional[dict]
    user_id:           str


# ── E-posta agent node'ları (async) ─────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)


async def draft_email_node(state: AgentState):
    """Kullanıcının talebine göre bir mail taslağı hazırlar."""
    last_msg = (state["messages"][-1].content
                if state.get("messages") else state["user_input"])

    system_prompt = (
        "Sen deneyimli bir akademik asistansın. Kullanıcının talebine göre profesyonel "
        "bir mail taslağı hazırla. KESİNLİKLE [İsminiz], [Hocanın Adı] gibi köşeli "
        "parantezli yer tutucular KULLANMA. Özel bilgi verilmediyse cümleyi bu boşlukları "
        "hissettirmeyecek şekilde dolaylı ve profesyonel kur. "
        "Cevabını SADECE şu JSON formatında ver: "
        '{"to": "mail_adresi", "subject": "konu", "body": "içerik"}'
    )

    response = await llm.ainvoke([
        ("system", system_prompt),
        ("human", f"Kullanıcı mesajı: {last_msg}"),
    ])

    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        draft = json.loads(content)
    except Exception:
        draft = {"to": "belirtilmemiş", "subject": "Akademik Konu", "body": response.content}

    presentation = (
        f"Taslak hazırlandı:\n\n"
        f"Kime: {draft.get('to')}\n"
        f"Konu: {draft.get('subject')}\n\n"
        f"İçerik: {draft.get('body')}\n\n"
        f"Bu maili yollamamı ister misin?"
    )

    return {
        "messages": [("assistant", presentation)],
        "draft_email": draft,
        "final_answer": presentation,
    }


async def send_email_node(state: AgentState):
    draft = state["draft_email"]
    result = await send_email_via_user_account.ainvoke({
        "to_email": draft["to"],
        "subject":  draft["subject"],
        "body":     draft["body"],
        "user_id":  state["user_id"],
    })
    return {"messages": [("assistant", result)], "final_answer": result}


# ── Yönlendirme ──────────────────────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    return {
        "selamlama": "selamlama_node",
        "kimlik":    "kimlik_node",
        "veda":      "veda_node",
        "hal_hatir": "hal_hatir_node",
        "yemekhane": "yemekhane_node",
        "mail":      "draft_email_node",
    }.get(state.get("intent", ""), "rag_node")


def route_after_rag(state: AgentState) -> str:
    return "llm_node" if state.get("intent") == "llm_sentez" else "dusuk_guven_node"


# ── Grafik ───────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)

for name, fn in [
    ("intent_node", intent_node), ("rag_node", rag_node), ("llm_node", llm_node),
    ("selamlama_node", selamlama_node), ("kimlik_node", kimlik_node),
    ("veda_node", veda_node), ("hal_hatir_node", hal_hatir_node),
    ("yemekhane_node", yemekhane_node), ("dusuk_guven_node", dusuk_guven_node),
    ("draft_email_node", draft_email_node),
    ("send_email_node", send_email_node),
]:
    workflow.add_node(name, fn)

workflow.set_entry_point("intent_node")

workflow.add_conditional_edges("intent_node", route_intent, {
    "selamlama_node": "selamlama_node",
    "kimlik_node":    "kimlik_node",
    "veda_node":      "veda_node",
    "hal_hatir_node": "hal_hatir_node",
    "yemekhane_node": "yemekhane_node",
    "draft_email_node": "draft_email_node",
    "rag_node":       "rag_node",
})

workflow.add_conditional_edges("rag_node", route_after_rag, {
    "llm_node":         "llm_node",
    "dusuk_guven_node": "dusuk_guven_node",
})

# E-posta akışı: taslak → [send_email_node'dan önce ONAY için interrupt] → gönder
workflow.add_edge("draft_email_node", "send_email_node")
workflow.add_edge("send_email_node", END)

# Uç node'lar
for n in ["llm_node", "selamlama_node", "kimlik_node", "veda_node",
          "hal_hatir_node", "yemekhane_node", "dusuk_guven_node"]:
    workflow.add_edge(n, END)
