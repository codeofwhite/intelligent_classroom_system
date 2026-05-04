"""
配置模块 - 存放所有常量和配置信息
"""
import os

# ========================
# 模型标签配置
# ========================
MODEL_CONFIGS = {
    "best_last_openvino_model": {
        "task": "detect",
        "labels_en": ["Raising-Hand", "Reading", "Writing", "Useing-Phone", "Head-down", "Sleep"],
        "labels_cn": ["举手", "看书", "写字", "使用手机", "低头做其他事情", "睡觉"],
        "focus": ["Raising-Hand", "Reading", "Writing"],
        "distract": ["Useing-Phone", "Head-down", "Sleep"],
    },
    "yolov8n-pose_openvino_model": {
        "task": "pose",
        "labels_en": ["normal posture", "raised hand", "looking down"],
        "labels_cn": ["正常坐姿", "举手", "低头"],
        "focus": ["normal posture", "raised hand"],
        "distract": ["looking down"],
    }
}

# ========================
# 数据库配置
# ========================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password123",
    "database": "user_center_db",
    "charset": "utf8mb4"
}

# ========================
# MinIO 配置
# ========================
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME = "video-bucket"

# ========================
# 目录配置
# ========================
MODELS_DIR = "models"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
KEY_FRAME_SAVE_DIR = "key_frames"

# ========================
# 视频源
# ========================
VIDEO_SOURCE = "http://192.168.26.157:8080/video"

# ========================
# 关键帧 / 统计参数
# ========================
GLOBAL_DISTRACT_NUM = 2
KEY_FRAME_INTERVAL = 30

os.makedirs(KEY_FRAME_SAVE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)