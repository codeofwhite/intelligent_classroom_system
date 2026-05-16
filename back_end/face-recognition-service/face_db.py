"""
人脸识别库 + 签到管理模块
- 人脸库加载（本地 / MinIO）
- 签到记录
"""
import os
import io
import tempfile
import threading
from datetime import datetime

import cv2
import face_recognition
from minio import Minio

from config import FACE_DIR, SIGN_LOG

# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS", "admin")
MINIO_SECRET = os.getenv("MINIO_SECRET", "")
FACE_BUCKET = "face-images"

# 全局人脸数据
known_encodings = []
known_names = []
signed_set = set()
lock = threading.Lock()


def load_face_database():
    """加载 faces/ 目录下所有人脸图片，提取编码"""
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)
        print("已创建 faces 文件夹，请放入图片")
        return

    for img_name in os.listdir(FACE_DIR):
        if img_name.lower().endswith(("jpg", "png", "jpeg")):
            img_path = os.path.join(FACE_DIR, img_name)
            _load_one_image(img_path, os.path.splitext(img_name)[0])

    print(f"✅ 本地加载完成：{len(known_names)} 人")


def load_face_database_from_minio():
    """从 MinIO face-images bucket 下载并加载所有人脸"""
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=False
    )

    objects = minio_client.list_objects(FACE_BUCKET, recursive=True)
    tmp_dir = tempfile.mkdtemp(prefix="face_cache_")
    count = 0

    for obj in objects:
        if not obj.object_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 从路径解析 student_code: "{class_code}/{student_code}/{filename}"
        parts = obj.object_name.split('/')
        if len(parts) < 3:
            continue
        student_code = parts[1]

        # 下载到临时目录
        local_path = os.path.join(tmp_dir, os.path.basename(obj.object_name))
        try:
            minio_client.fget_object(FACE_BUCKET, obj.object_name, local_path)
            _load_one_image(local_path, student_code)
            count += 1
        except Exception as e:
            print(f"下载/加载失败 {obj.object_name}: {e}")

    print(f"✅ MinIO 加载完成：{len(known_names)} 人（{count} 张照片）")
    return tmp_dir


def _load_one_image(img_path, name):
    """从单张图片中提取人脸编码并追加到全局列表"""
    img = cv2.imread(img_path)
    if img is None:
        return
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    known_encodings.append(face_encoding)
    known_names.append(name)


def sign_in(name):
    """
    签到（线程安全，每人只签一次）
    返回 True 表示本次签到成功，False 表示已签到过
    """
    with lock:
        if name in signed_set:
            return False

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SIGN_LOG, "a+", encoding="utf-8") as f:
            f.write(f"{name} {time_str}\n")
        signed_set.add(name)
        print(f"✅ {name} 签到成功！")
        return True


def get_sign_logs():
    """读取签到日志"""
    try:
        with open(SIGN_LOG, 'r', encoding='utf-8') as f:
            logs = f.readlines()
        return logs
    except FileNotFoundError:
        return []