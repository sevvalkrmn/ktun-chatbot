from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    intent_node,
    rag_node,
    llm_node,
    selamlama_node,
    kimlik_node,
    dusuk_guven_node,
    konu_disi_node,
    kufur_node,
    veda_node,
    hal_hatir_node,
    anlamsiz_node
)

# ── KARAR FONKSİYONU ─────────────────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "")

    if intent == "selamlama":
        return "selamlama_node"
    elif intent == "kimlik":
        return "kimlik_node"
    elif intent == "konu_disi":
        return "konu_disi_node"
    elif intent == "kufur":
        return "kufur_node"
    elif intent == "veda":
        return "veda_node"
    elif intent == "hal_hatir":
        return "hal_hatir_node"
    elif intent == "anlamsiz":
        return "anlamsiz_node"
    elif intent == "takvim_ekle":
        return "rag_node"
    else:
        return "rag_node"       # Akademik soru — RAG'a git


def route_after_rag(state: AgentState) -> str:
    """
    rag_node'dan sonra nereye gidileceğine karar verir.
    Güven skoruna göre yönlendirir.
    """
    intent = state.get("intent", "")

    if intent == "dogrudan_cevap":
        return END                  # Cevap hazır, bitir
    elif intent == "llm_sentez":
        return "llm_node"           # LLM sentez yapsın
    else:
        return "dusuk_guven_node"   # Bilmiyorum


# ── GRAPH KURULUMU ───────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Node'ları ekle
    graph.add_node("intent_node",      intent_node)
    graph.add_node("rag_node",         rag_node)
    graph.add_node("llm_node",         llm_node)
    graph.add_node("selamlama_node",   selamlama_node)
    graph.add_node("kimlik_node",      kimlik_node)
    graph.add_node("dusuk_guven_node", dusuk_guven_node)
    graph.add_node("konu_disi_node",   konu_disi_node)
    graph.add_node("kufur_node",    kufur_node)
    graph.add_node("veda_node",     veda_node)
    graph.add_node("hal_hatir_node", hal_hatir_node)
    graph.add_node("anlamsiz_node", anlamsiz_node)

    # Başlangıç noktası
    graph.set_entry_point("intent_node")

    # intent_node'dan sonra koşullu yönlendirme
    graph.add_conditional_edges(
    "intent_node",
    route_intent,
    {
        "selamlama_node":  "selamlama_node",
        "kimlik_node":     "kimlik_node",
        "konu_disi_node":  "konu_disi_node",
        "kufur_node":      "kufur_node",
        "veda_node":       "veda_node",
        "hal_hatir_node":  "hal_hatir_node",
        "anlamsiz_node":   "anlamsiz_node",
        "rag_node":        "rag_node"
    }
)

    # rag_node'dan sonra koşullu yönlendirme
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {
            END:                END,
            "llm_node":         "llm_node",
            "dusuk_guven_node": "dusuk_guven_node"
        }
    )

    # Bu node'lardan sonra her zaman bitir
    graph.add_edge("llm_node",         END)
    graph.add_edge("selamlama_node",   END)
    graph.add_edge("kimlik_node",      END)
    graph.add_edge("dusuk_guven_node", END)
    graph.add_edge("konu_disi_node",   END)
    graph.add_edge("kufur_node",     END)
    graph.add_edge("veda_node",      END)
    graph.add_edge("hal_hatir_node", END)
    graph.add_edge("anlamsiz_node",  END)

    return graph.compile()
