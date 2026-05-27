import asyncpg
from config import Config

_pool = None


async def init_db():
    global _pool
    db = Config.DatabaseConfig
    _pool = await asyncpg.create_pool(
        host=db.DB_HOST,
        port=db.DB_PORT,
        database=db.DB_NAME,
        user=db.DB_USER,
        password=db.DB_PASSWORD,
        min_size=2,
        max_size=10,
    )


async def close_db():
    global _pool
    if _pool:
        await _pool.close()


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("数据库连接池未初始化，请先调用 init_db()")
    return _pool
