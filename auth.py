import jwt
from datetime import datetime, timedelta, timezone
from config import Config
from database import get_pool


async def verify_user(account: str, password: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, description FROM users WHERE account = $1 AND password = $2",
            account,
            password,
        )
    if row is None:
        return None
    return {"id": row["id"], "description": row["description"]}


def create_token(user_id: int, description: str) -> str:
    j = Config.JWTConfig
    payload = {
        "id": user_id,
        "description": description,
        "exp": datetime.now(timezone.utc) + timedelta(hours=j.EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, j.SECRET, algorithm=j.ALGORITHM)
