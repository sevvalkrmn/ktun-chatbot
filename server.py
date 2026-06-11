"""
KTÜN Asistanı — FastAPI sunucusu (agent + OAuth + onaylı e-posta).

Çalıştırma:  python server.py   (veya: uvicorn server:app --port 8000)
Gmail bağlamak için tarayıcıdan: http://localhost:8000/auth/login
"""
import os
import sys

# Windows konsolunda Türkçe karakterli print'ler için (uvicorn alt süreci dahil)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from uuid import uuid4
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))

from db import async_session, init_db
from db_models import UserOAuth
from oauth_config import oauth
from agent_graph import workflow
from rag.resources import get_resources
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB = os.path.join(BASE, "checkpoints.db")

# Lifespan boyunca yaşayan nesneler
GRAPH = None
_saver_cm = None


def _br(text):
    """Satır sonlarını arayüzde görünür kıl."""
    return (text or "").replace("\n", "<br>")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GRAPH, _saver_cm
    # RAG kaynaklarını önceden ısıt
    get_resources()
    # OAuth token tablosunu oluştur
    await init_db()
    # SQLite checkpointer (onay/resume akışı için)
    _saver_cm = AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB)
    saver = await _saver_cm.__aenter__()
    await saver.setup()
    GRAPH = workflow.compile(checkpointer=saver,
                             interrupt_before=["send_email_node"])
    print("Sunucu hazır.")
    yield
    if _saver_cm:
        await _saver_cm.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware,
                   secret_key=os.getenv("SECRET_KEY", "dev-secret-key"))
app.mount("/static",
          StaticFiles(directory=os.path.join(BASE, "arayüz", "static")),
          name="static")
templates = Jinja2Templates(directory=os.path.join(BASE, "arayüz", "templates"))


# ── UI ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html")


# ── Google OAuth ─────────────────────────────────────────────────────────
@app.get("/auth/login")
async def login(request: Request):
    redirect_uri = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    async with async_session() as session:
        db_user = UserOAuth(
            user_id=user_info["email"],
            access_token=token["access_token"],
            refresh_token=token.get("refresh_token"),
            token_expiry=datetime.utcnow() + timedelta(seconds=token.get("expires_in", 3600)),
            provider="google",
        )
        await session.merge(db_user)
        await session.commit()
    request.session["user_email"] = user_info["email"]
    return RedirectResponse(url="/chat")


# ── Sohbet ───────────────────────────────────────────────────────────────
@app.post("/api/ask")
@app.post("/chat/ask")
async def ask_chatbot(request: Request):
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"response": "Lütfen bir soru yazın."})

    user_id = request.session.get("user_email", "test@example.com")
    thread_id = str(uuid4())          # her soru için taze thread
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages":   [("user", message)],
        "user_input": message,
        "user_id":    user_id,
        "intent":     "",
        "final_answer": "",
        "draft_email": None,
    }

    last_event = None
    async for event in GRAPH.astream(initial_state, config=config, stream_mode="updates"):
        last_event = event
        if "draft_email_node" in event:
            nd = event["draft_email_node"]
            # onay akışı: bu thread'i sakla, resume bekleniyor
            request.session["pending_thread"] = thread_id
            return JSONResponse({
                "response": _br(nd.get("final_answer")),
                "durum":    "ONAY BEKLİYOR",
                "taslak":   nd.get("draft_email"),
            })

    if last_event:
        for nd in last_event.values():
            if isinstance(nd, dict) and nd.get("final_answer"):
                return JSONResponse({"response": _br(nd["final_answer"]),
                                     "durum": "TAMAMLANDI"})

    return JSONResponse({"response": "Üzgünüm, bir cevap oluşturamadım."})


@app.post("/chat/resume")
async def resume_chat(request: Request):
    data = await request.json()
    choice = (data.get("user_choice") or "hayır").lower()
    thread_id = request.session.pop("pending_thread", None)

    if not thread_id:
        return JSONResponse({"response": "Onay bekleyen bir işlem bulunamadı."})

    config = {"configurable": {"thread_id": thread_id}}

    if choice in ("evet", "onayla", "gönder"):
        async for event in GRAPH.astream(None, config=config, stream_mode="updates"):
            if "send_email_node" in event:
                return JSONResponse({"response": _br(
                    event["send_email_node"].get("final_answer", "Mail gönderildi."))})
        return JSONResponse({"response": "İşlem tamamlandı."})

    return JSONResponse({"response": "Mail gönderimi iptal edildi."})


if __name__ == "__main__":
    import uvicorn
    print("\n>>> Tarayıcıda aç: http://localhost:8000\n")
    # 127.0.0.1 = sadece bu bilgisayar. Aynı ağdaki telefon/başka cihazdan
    # erişmek istersen host="0.0.0.0" yap ve http://<bilgisayar-IP>:8000 kullan.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
