import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import Config
from database import get_db

security = HTTPBearer()


async def verify_user(account: str, password: str):
    db = get_db()
    cursor = await db.execute(
        "SELECT id, account, description FROM users WHERE account = ? AND password = ?",
        (account, password),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {"id": row[0], "account": row[1], "description": row[2]}


def create_token(user_id: int, account: str, description: str) -> str:
    j = Config.JWTConfig
    payload = {
        "id": user_id,
        "account": account,
        "description": description,
        "exp": datetime.now(timezone.utc) + timedelta(hours=j.EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, j.SECRET, algorithm=j.ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """JWT 鉴权依赖 — 从 Authorization Header 提取并验证 Token，返回 payload"""
    token = credentials.credentials
    j = Config.JWTConfig
    try:
        payload = jwt.decode(token, j.SECRET, algorithms=[j.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的令牌")
