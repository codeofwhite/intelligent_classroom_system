"""
共享状态模块 - 存放全局可变状态和共享客户端实例
所有蓝图通过导入此模块来访问共享资源
"""
import os
import sys
import pymysql
from minio import Minio
from ultralytics import YOLO
from collections import deque

from config import (
    MODEL_CONFIGS, DB_CONFIG, MINIO_ENDPOINT, MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY, BUCKET_NAME, MODELS_DIR, VIDEO_SOURCE,
    KEY_FRAME_SAVE_DIR
)

# ========================
# ByteTrack
# ========================
sys.path.append('./ByteTrack')
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


class ByteTrackArgs:
    track_thresh = 0.2
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False


# ========================
# 模型实例
# ========================
available_models = list(MODEL_CONFIGS.keys())
current_model_name = available_models[0] if available_models else None
current_model_config = MODEL_CONFIGS[current_model_name] if current_model_name else None
model = None

if current_model_name:
    model = YOLO(
        os.path.join(MODELS_DIR, current_model_name),
        task=current_model_config['task']
    )

# 覆盖为固定模型（保持原逻辑）
model = YOLO("models/best_last_openvino_model/", task='detect')

# ========================
# MinIO 客户端
# ========================
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# ========================
# 数据库连接（供非请求上下文使用）
# ========================
db = pymysql.connect(
    host=DB_CONFIG["host"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    database=DB_CONFIG["database"],
    charset=DB_CONFIG["charset"]
)


def get_db_connection():
    """每次请求新建连接，用完关闭"""
    return pymysql.connect(**DB_CONFIG)


# ========================
# 视频源
# ========================
video_source = VIDEO_SOURCE

# ========================
# 关键帧参数
# ========================
last_capture_frame = 0

# ========================
# 实时监测全局状态
# ========================
is_recording = False
real_time_stats = {
    "hand_up": 0,       # Raising-Hand 举手
    "study_norm": 0,    # Reading / Writing 正常学习
    "look_down": 0,     # Head-down 低头
    "abnormal": 0       # Useing-Phone / Sleep 严重分心
}
real_time_logs = []
video_writer = None
output_video_path = None