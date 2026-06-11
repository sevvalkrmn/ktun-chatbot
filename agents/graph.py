from langgraph.graph import StateGraph, END
from agents.state import AgentState
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

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "")

    if intent == "selamlama":
        return "selamlama_node"
    elif intent == "kimlik":
        return "kimlik_node"
    elif intent == "veda":
        return "veda_node"
    elif intent == "hal_hatir":
        return "hal_hatir_node"
    elif intent == "yemekhane":
        return "yemekhane_node"
    else:
        return "rag_node"  # Geri kalan her şey RAG'a

def route_after_rag(state: AgentState) -> str:
    intent = state.get("intent", "")

    if intent == "llm_sentez":
        return "llm_node"
    else:
        return "dusuk_guven_node"

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Node'ları ekle
    graph.add_node("intent_node",      intent_node)
    graph.add_node("rag_node",         rag_node)
    graph.add_node("llm_node",         llm_node)
    graph.add_node("selamlama_node",   selamlama_node)
    graph.add_node("kimlik_node",      kimlik_node)
    graph.add_node("veda_node",        veda_node)
    graph.add_node("hal_hatir_node",   hal_hatir_node)
    graph.add_node("yemekhane_node",   yemekhane_node)
    graph.add_node("dusuk_guven_node", dusuk_guven_node)

    # Başlangıç noktası
    graph.set_entry_point("intent_node")

    # intent_node'dan sonra yönlendirme
    graph.add_conditional_edges(
        "intent_node",
        route_intent,
        {
            "selamlama_node": "selamlama_node",
            "kimlik_node":    "kimlik_node",
            "veda_node":      "veda_node",
            "hal_hatir_node": "hal_hatir_node",
            "yemekhane_node": "yemekhane_node",
            "rag_node":       "rag_node"
        }
    )

    # rag_node'dan sonra yönlendirme
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {
            "llm_node":         "llm_node",
            "dusuk_guven_node": "dusuk_guven_node"
        }
    )

    # Tüm son node'lardan END'e
    graph.add_edge("llm_node",         END)
    graph.add_edge("selamlama_node",   END)
    graph.add_edge("kimlik_node",      END)
    graph.add_edge("veda_node",        END)
    graph.add_edge("hal_hatir_node",   END)
    graph.add_edge("yemekhane_node",   END)
    graph.add_edge("dusuk_guven_node", END)

    return graph.compile()