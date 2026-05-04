"""
人脸识别库 + 签到管理模块
- 人脸库加载
- 签到记录
"""
import os
import threading
from datetime import datetime

import cv2
import face_recognition

from config import FACE_DIR, SIGN_LOG

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
            img = cv2.imread(img_path)
            if img is None:
                print(f"跳过无效图片：{img_name}")
                continue
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                print(f"未检测到人脸：{img_name}")
                continue

            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_encodings.append(face_encoding)
            known_names.append(os.path.splitext(img_name)[0])

    print(f"✅ 加载完成：{len(known_names)} 人")


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