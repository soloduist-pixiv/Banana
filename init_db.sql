CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    account     VARCHAR(255) UNIQUE NOT NULL,
    password    VARCHAR(255) NOT NULL,
    description VARCHAR(255) DEFAULT ''
);

-- 插入一条测试数据（密码明文）
INSERT INTO users (account, password, description)
VALUES ('admin', '123456', '管理员')
ON CONFLICT (account) DO NOTHING;
