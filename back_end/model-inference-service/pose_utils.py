# pose_utils.py
import face_recognition
import os
import numpy as np

FACE_LIB = "face_lib"
POSE_CONF = 0.5
known_encodings = []
known_ids = []

def load_face_database():
    global known_encodings, known_ids
    known_encodings = []
    known_ids = []

    if not os.path.exists(FACE_LIB):
        os.makedirs(FACE_LIB)
        return

    for student_id in os.listdir(FACE_LIB):
        stu_dir = os.path.join(FACE_LIB, student_id)
        if not os.path.isdir(stu_dir):
            continue

        for fname in os.listdir(stu_dir):
            if not fname.endswith(('jpg', 'png', 'jpeg')):
                continue

            img_path = os.path.join(stu_dir, fname)
            img = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_ids.append(student_id)

def get_behavior(keypoints):
    nose_y = keypoints[0, 1] if keypoints[0, 2] > 0.5 else 9999
    ls_y = keypoints[5, 1] if keypoints[5, 2] > 0.5 else 9999
    rs_y = keypoints[6, 1] if keypoints[6, 2] > 0.5 else 9999
    le_y = keypoints[9, 1] if keypoints[9, 2] > 0.5 else 9999
    re_y = keypoints[10, 1] if keypoints[10, 2] > 0.5 else 9999

    avg_shoulder = (ls_y + rs_y) / 2

    if re_y < avg_shoulder - 15 or le_y < avg_shoulder - 15:
        return "raised hand"
    if nose_y > avg_shoulder - 10:
        return "looking down"
    return "normal posture"

# 启动时自动加载人脸库
load_face_database()