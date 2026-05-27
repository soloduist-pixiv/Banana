import os

class Config:
    # 基础配置
    UPLOAD_FOLDER = 'uploads'
    IMAGE_FOLDER = 'image'
    MODEL_PATH = '5个模型的训练结果/shufflenet_v2/best_model.pth'
    SECRET_KEY = 'your-secret-key'

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