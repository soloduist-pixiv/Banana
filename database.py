import aiosqlite
from config import Config

_db = None


async def init_db():
    global _db
    db_path = Config.DatabaseConfig.DB_PATH
    _db = await aiosqlite.connect(db_path)
    _db.row_factory = aiosqlite.Row

    # 创建 users 表
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            account     TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            description TEXT DEFAULT ''
        )
    """)

    # 插入测试用户 (admin / 123456)
    await _db.execute("""
        INSERT OR IGNORE INTO users (account, password, description)
        VALUES ('admin', '123456', '管理员')
    """)
    await _db.commit()


async def close_db():
    global _db
    if _db:
        await _db.close()
        _db = None


def get_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    return _db
