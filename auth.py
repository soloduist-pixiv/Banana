import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import Config
from database import get_pool

security = HTTPBearer()


async def verify_user(account: str, password: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, account, description FROM users WHERE account = $1 AND password = $2",
            account,
            password,
        )
    if row is None:
        return None
    return {"id": row["id"], "account": row["account"], "description": row["description"]}


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
