import os
import sys

# Windows konsolunda Türkçe karakterli print'ler için UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Proje kökünü path'e ekle (arayüz/ alt klasöründen çalışıyoruz) ───────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# config import'undan ÖNCE .env yüklenmeli (OPENAI_API_KEY oradan okunuyor)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from flask import Flask, render_template, request, jsonify

from rag.embedder import BGE_M3_Embedder
from agents.graph import build_graph
from rag.retriever import load_data, build_corpus, build_bm25, build_faiss
from config import EMBEDDING_MODEL, DATA_PATH

app = Flask(__name__)

# ── RAG kaynakları: süreç boyunca BİR KEZ yüklenir, sıcak tutulur ────────
_RESOURCES = {}


def get_resources():
    """bge-m3 + FAISS + BM25 + graph'ı bir kez kurar, sonraki çağrılarda cache'ten döner."""
    if not _RESOURCES:
        print("RAG kaynakları yükleniyor (ilk açılış, ~30-40 sn)...")
        data = load_data(os.path.join(ROOT, DATA_PATH))
        corpus_texts, corpus_answers = build_corpus(data)
        embedding_model = BGE_M3_Embedder(EMBEDDING_MODEL)
        _RESOURCES.update(
            bm25            = build_bm25(corpus_texts),
            embedding_model = embedding_model,
            faiss_index     = build_faiss(corpus_texts, embedding_model),
            corpus_texts    = corpus_texts,
            corpus_answers  = corpus_answers,
            graph           = build_graph(),
        )
        print("RAG hazır.")
    return _RESOURCES


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


# ── Asıl API endpoint'i — RAG + agent hattına bağlandı ───────────────────
@app.route('/api/ask', methods=['POST'])
def ask_aila():
    data = request.json or {}
    user_message = (data.get('message') or '').strip()

    if not user_message:
        return jsonify({"response": "Lütfen bir soru yazın."})

    res = get_resources()
    state = {
        "user_input":        user_message,
        "intent":            "",
        "retrieved_context": "",
        "confidence_score":  0.0,
        "final_answer":      "",
        "error":             "",
        "bm25":              res["bm25"],
        "faiss_index":       res["faiss_index"],
        "corpus_texts":      res["corpus_texts"],
        "corpus_answers":    res["corpus_answers"],
        "embedding_model":   res["embedding_model"],
        "retrieval_query":   "",
    }

    try:
        result = res["graph"].invoke(state)
        answer = result.get("final_answer", "Bir hata oluştu.")
    except Exception as e:
        print(f"API hatası: {e}")
        answer = "Bir hata oluştu, lütfen tekrar deneyin."

    # Satır sonlarını arayüzde görünür kıl
    answer_html = answer.replace("\n", "<br>")
    return jsonify({"response": answer_html})


if __name__ == '__main__':
    # Sunucu açılır açılmaz kaynakları yükle ki ilk kullanıcı beklemesin
    get_resources()
    # use_reloader=False → debug modda modelin iki kez yüklenmesini önler
    app.run(debug=True, use_reloader=False, port=5000)
