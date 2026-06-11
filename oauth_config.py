"""Google OAuth (authlib + Starlette/FastAPI) yapılandırması."""
import os
from dotenv import load_dotenv
from authlib.integrations.starlette_client import OAuth

load_dotenv()

oauth = OAuth()

CONF_URL_GOOGLE = "https://accounts.google.com/.well-known/openid-configuration"

oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url=CONF_URL_GOOGLE,
    client_kwargs={
        # gmail.send → kullanıcı adına e-posta gönderme yetkisi
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.send",
        # refresh_token alabilmek için
        "access_type": "offline",
        "prompt": "consent",
    },
)
