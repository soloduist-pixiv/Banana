import os

class Config:
    # 基础配置
    UPLOAD_FOLDER = 'uploads'
    IMAGE_FOLDER = 'image'
    MODEL_PATH = '5个模型的训练结果/shufflenet_v2/best_model.pth'
    SECRET_KEY = 'your-secret-key'

    # 数据库配置 — 部署时修改
    class DatabaseConfig:
        DB_HOST = "your-server-ip"
        DB_PORT = 5432
        DB_NAME = "banana"
        DB_USER = "postgres"
        DB_PASSWORD = "your-password"

    # JWT 配置
    class JWTConfig:
        SECRET = "your-jwt-secret"
        ALGORITHM = "HS256"
        EXPIRE_HOURS = 24

    # 确保上传目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

class DevelopmentConfig(Config):
    # 开发环境配置
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 8080

class ProductionConfig(Config):
    # 生产环境配置
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 80
