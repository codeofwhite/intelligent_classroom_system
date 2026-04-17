import face_recognition
import cv2
import numpy as np
from datetime import datetime
import os
from flask import Flask, render_template, Response, jsonify
import threading

# ===================== Flask 应用 =====================
app = Flask(__name__)

# ===================== 配置 =====================
SIGN_LOG = "sign_log.txt"
FACE_DIR = "faces/"

# 全局变量
known_encodings = []
known_names = []
signed_set = set()
lock = threading.Lock()

# ===================== 加载人脸库 =====================
def load_face_database():
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
                continue
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                continue

            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_encodings.append(face_encoding)
            known_names.append(os.path.splitext(img_name)[0])

    print(f"✅ 加载完成：{len(known_names)} 人")

# ===================== 签到记录 =====================
def sign_in(name):
    with lock:
        if name in signed_set:
            return False

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SIGN_LOG, "a+", encoding="utf-8") as f:
            f.write(f"{name} {time_str}\n")
        signed_set.add(name)
        print(f"✅ {name} 签到成功！")
        return True

# ===================== 视频流生成 =====================
def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "未知"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                sign_in(name)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ===================== 接口 =====================
@app.route('/')
def index():
    return "人脸识别签到服务已启动"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sign_log')
def get_sign_log():
    try:
        with open(SIGN_LOG, 'r', encoding='utf-8') as f:
            logs = f.readlines()
        return jsonify(logs=logs)
    except:
        return jsonify(logs=[])

# ===================== 启动 =====================
if __name__ == "__main__":
    load_face_database()
    app.run(host="0.0.0.0", port=5001, debug=True)