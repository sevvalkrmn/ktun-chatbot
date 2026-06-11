"""Agent araçları — kullanıcının Gmail hesabından e-posta gönderme."""
import base64
from email.mime.text import MIMEText

import httpx
from langchain_core.tools import tool
from sqlalchemy import select

from db_models import UserOAuth
from db import async_session


async def get_user_token(user_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(UserOAuth).where(UserOAuth.user_id == user_id)
        )
        return result.scalars().first()


@tool
async def send_email_via_user_account(to_email: str, subject: str, body: str, user_id: str):
    """
    Kullanıcının kendi Google (Gmail) hesabı üzerinden e-posta gönderir.
    - to_email: Alıcının e-posta adresi
    - subject:  E-posta konusu
    - body:     E-posta içeriği
    - user_id:  Gönderen kullanıcının kimliği (e-posta adresi)
    """
    user_data = await get_user_token(user_id)
    if not user_data:
        return ("Hata: Google hesabı bağlantısı bulunamadı. "
                "Lütfen önce /auth/login ile Gmail hesabınızı bağlayın.")

    url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    message = MIMEText(body)
    message["to"] = to_email
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    headers = {
        "Authorization": f"Bearer {user_data.access_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json={"raw": raw}, headers=headers)
            if resp.status_code == 200:
                return f"Mail başarıyla '{to_email}' adresine gönderildi."
            return f"Hata: Gmail API hatası ({resp.status_code}). Token süresi dolmuş olabilir, tekrar giriş yapın."
        except Exception as e:
            return f"Hata: Bağlantı hatası - {str(e)}"
