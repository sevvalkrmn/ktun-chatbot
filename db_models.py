from sqlalchemy import Column, String, DateTime, Text, Integer
from db import Base


class UserOAuth(Base):
    __tablename__ = "user_oauth"

    id           = Column(Integer, primary_key=True)
    user_id      = Column(String, unique=True, index=True)  # kullanıcı e-postası
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    token_expiry = Column(DateTime, nullable=False)
    provider     = Column(String(20), default="google")
