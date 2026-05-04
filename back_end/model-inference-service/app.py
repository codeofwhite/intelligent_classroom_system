# 系统基础库
import os
import sys
import csv
import json
import time
import io
import tempfile
from datetime import date, datetime
from collections import deque

# 科学计算 / 图像处理
import numpy as np
import cv2
import face_recognition

# AI 模型
from ultralytics import YOLO
from openai import OpenAI

# 数据库 / 存储
import pymysql
from minio import Minio

# Flask 服务
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

# 自定义工具模块
from pose_utils import get_behavior, load_face_database, known_encodings, known_ids, POSE_CONF
from ai_agent import analyze_class_report
from chat_agent import chat_agent_api, get_session_messages, get_teacher_sessions

# ========================
# 🔌 可扩展模型 + 标签配置（核心！）
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

# 当前模型配置（自动跟随切换）
current_model_config = None

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password123",
    "database": "user_center_db",
    "charset": "utf8mb4"
}

db = pymysql.connect(
    host="localhost",
    user="root",
    password="password123",
    database="user_center_db",
    charset='utf8mb4'
)

GLOBAL_DISTRACT_NUM = 2
KEY_FRAME_INTERVAL = 30
KEY_FRAME_SAVE_DIR = "key_frames"
os.makedirs(KEY_FRAME_SAVE_DIR, exist_ok=True)

last_capture_frame = 0

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

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

MODELS_DIR = "models"
available_models = list(MODEL_CONFIGS.keys())  # 从配置里读，更干净
current_model_name = available_models[0] if available_models else None
model = None
current_model_config = None

if current_model_name:
    current_model_config = MODEL_CONFIGS[current_model_name]
    model = YOLO(
        os.path.join(MODELS_DIR, current_model_name),
        task=current_model_config['task']
    )

minio_client = Minio(
    "minio:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)
BUCKET_NAME = "video-bucket"

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("models/best_last_openvino_model/", task='detect')
video_source = "http://192.168.26.157:8080/video"

# ========================
# 颜色函数
# ========================
def get_color(category_id):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return colors[category_id % len(colors)]

# ==========================
# 实时监测 全局状态
# ==========================
is_recording = False
real_time_stats = {
    "hand_up": 0,      # Raising-Hand 举手
    "study_norm": 0,   # Reading / Writing 正常学习
    "look_down": 0,    # Head-down 低头
    "abnormal": 0      # Useing-Phone / Sleep 严重分心
}
real_time_logs = []
video_writer = None
output_video_path = None

# ==========================
# 开始录制
# ==========================
@app.route('/start_record', methods=['POST'])
def start_record():
    global is_recording, video_writer, output_video_path
    if is_recording:
        return jsonify({"status": "already recording"})
    
    data = request.json
    teacher_code = data.get("teacher_code", "T2025001")
    class_code = data.get("class_code", 1)
    lesson_section = data.get("lesson_section", "实时课堂")

    # 初始化视频保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = tempfile.gettempdir()
    output_video_path = os.path.join(tmp_dir, f"record_{timestamp}.mp4")
    
    # 获取分辨率
    cap = cv2.VideoCapture(video_source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    cap.release()

    # 清空统计
    real_time_stats["hand_up"] = 0
    real_time_stats["study_norm"] = 0
    real_time_stats["look_down"] = 0
    real_time_stats["abnormal"] = 0
    real_time_logs.clear()

    is_recording = True
    return jsonify({"status": "started"})

# ==========================
# 停止录制 + 自动上传入库
# ==========================
@app.route('/stop_record', methods=['POST'])
def stop_record():
    global is_recording, video_writer
    if not is_recording:
        return jsonify({"status": "not recording"})

    is_recording = False
    if video_writer:
        video_writer.release()

    data = request.json
    teacher_code = data.get("teacher_code", "T2025001")
    class_code = data.get("class_code", 1)
    lesson_section = data.get("lesson_section", "实时课堂")

    # 自动上传逻辑（和你 upload 接口一模一样）
    try:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"teacher_{teacher_code}/class_{class_code}/{time_str}_{lesson_section}"
        video_obj = f"{base}/live.mp4"
        json_obj = f"{base}/live_stats.json"

        # 保存统计
        stats = {
            "total_frames": 0,
            "behavior_counts": {
                "举手": real_time_stats["hand_up"],
                "看书": real_time_stats["study_norm"],
                "写字": 0,
                "使用手机": real_time_stats["abnormal"],
                "低头做其他事情": real_time_stats["look_down"],
                "睡觉": 0
            }
        }
        json_path = os.path.join(tempfile.gettempdir(), "live_stats.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)

        # 上传 MinIO
        minio_client.fput_object(BUCKET_NAME, video_obj, output_video_path)
        minio_client.fput_object(BUCKET_NAME, json_obj, json_path)

        # 写入数据库
        db_temp = pymysql.connect(
            host="localhost", user="root", password="password123", database="user_center_db", charset='utf8mb4'
        )
        cursor = db_temp.cursor()
        report_code = f"R{datetime.now().strftime('%Y%m%d%H%M%S')}"
        cursor.execute("""
            INSERT INTO course_reports
            (report_code, teacher_code, class_code, lesson_section, minio_video_path, minio_json_path, minio_csv_path)
            VALUES (%s,%s,%s,%s,%s,%s,'')
        """, (report_code, teacher_code, int(class_code), lesson_section, video_obj, json_obj))
        db_temp.commit()
        cursor.close()
        db_temp.close()

        return jsonify({"status": "stopped & saved & uploaded"})
    except Exception as e:
        print("SAVE ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ==========================
# 获取实时统计
# ==========================
@app.route('/get_realtime_stats', methods=['GET'])
def get_realtime_stats():
    # 计算实时专注率
    total = real_time_stats["hand_up"] + real_time_stats["study_norm"] + real_time_stats["look_down"] + real_time_stats["abnormal"]
    focus_rate = 100 * (real_time_stats["hand_up"] + real_time_stats["study_norm"]) / total if total > 0 else 0

    return jsonify({
        "stats": real_time_stats,
        "focus_rate": round(focus_rate, 1),
        "logs": real_time_logs[-10:]
    })

# ==========================
# 获取录制状态
# ==========================
@app.route('/get_record_status', methods=['GET'])
def get_record_status():
    return jsonify({"recording": is_recording})

# ========================
# 实时流
# ========================
def generate_frames():
    global last_capture_frame, video_writer
    camera = cv2.VideoCapture(video_source)
    tracker = BYTETracker(ByteTrackArgs())
    track_history = {}

    while True:
        success, frame = camera.read()
        if not success:
            break

        # 保存视频（如果正在录制）
        if is_recording and video_writer:
            video_writer.write(frame)

        results = model.predict(frame, imgsz=640, verbose=False, half=True, conf=0.25)
        det = results[0].boxes.data.cpu().numpy()

        # 实时统计临时变量
        rt_hand = 0
        rt_study_norm = 0
        rt_look_down = 0
        rt_abnormal = 0

        if len(det) > 0:
            online_targets = tracker.update(det[:, :5], [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
            for t in online_targets:
                tlbr = t.tlbr
                tid = t.track_id
                x1, y1, x2, y2 = map(int, tlbr)

                best_cls = 0
                min_dist = 1e9
                for d in det:
                    cx = (d[0]+d[2])/2
                    cy = (d[1]+d[3])/2
                    tx = (x1+x2)/2
                    ty = (y1+y2)/2
                    dist = (tx-cx)**2 + (ty-cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_cls = int(d[5])

                label = results[0].names[best_cls]
                if tid not in track_history:
                    track_history[tid] = deque(maxlen=15)
                track_history[tid].append(label)
                final_label = max(set(track_history[tid]), key=track_history[tid].count)

                color = get_color(best_cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID{tid} {final_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # ======================
                # 实时行为统计
                # ======================
                if final_label == "Raising-Hand":
                    rt_hand += 1
                elif final_label in ("Reading", "Writing"):
                    rt_study_norm += 1
                elif final_label == "Head-down":
                    rt_look_down += 1
                elif final_label in ("Useing-Phone", "Sleep"):
                    rt_abnormal += 1

        # 更新全局实时数据
        if is_recording:
            real_time_stats["hand_up"] = rt_hand
            real_time_stats["study_norm"] = rt_study_norm
            real_time_stats["look_down"] = rt_look_down
            real_time_stats["abnormal"] = rt_abnormal

            # 日志
            now = time.strftime("%H:%M:%S")
            log = f"[{now}] 举手:{rt_hand} 抬头:{rt_study_norm} 低头:{rt_look_down}"
            real_time_logs.append(log)

        frame_display = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_models', methods=['GET'])
def get_models():
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return jsonify({"models": models, "current": current_model_name})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global model, current_model_name, current_model_config
    target_model = request.json.get('model_name')
    
    if target_model not in MODEL_CONFIGS:
        return jsonify({"status": "error", "msg": "模型未配置，请先在 MODEL_CONFIGS 中定义"}), 404
    
    try:
        cfg = MODEL_CONFIGS[target_model]
        model_path = os.path.join(MODELS_DIR, target_model)
        model = YOLO(model_path, task=cfg['task'])
        
        current_model_name = target_model
        current_model_config = cfg  # 自动绑定标签配置
        
        return jsonify({
            "status": "success",
            "current": current_model_name,
            "labels": cfg['labels_cn']
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

# ========================
# ✅ 修复完成的上传分析
# ========================
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    teacher_code = request.form.get('teacher_code')
    class_code = request.form.get('class_code')
    lesson_section = request.form.get('lesson_section')
    file = request.files['video']

    face_id_cache = set()

    # 动态从当前模型配置获取标签
    if current_model_config is None:
        return jsonify({"error": "No model selected"}), 400

    CLASS_NAMES_EN = current_model_config["labels_en"]
    CLASS_NAMES_CN = current_model_config["labels_cn"]
    FOCUS_BEHAVIOR = current_model_config["focus"]
    DISTRACT_BEHAVIOR = current_model_config["distract"]
    task_type = current_model_config["task"]  # detect or pose

    total_count = {cls: 0 for cls in CLASS_NAMES_CN}
    student_behaviors = {}
    frame_count = 0
    behavior_data = []

    # 改用项目目录下的临时文件夹，彻底解决 WinError 267
    base_tmp = "./tmp_upload"
    os.makedirs(base_tmp, exist_ok=True)

    input_path = os.path.join(base_tmp, "input.mp4")
    output_path = os.path.join(base_tmp, "output.mp4")
    csv_path = os.path.join(base_tmp, "tracks.csv")
    json_path = os.path.join(base_tmp, "result.json")

    try:
        # 强制保存 + 确保文件写入
        file.save(input_path)
        import time
        time.sleep(0.1)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = BYTETracker(ByteTrackArgs())
        track_history = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ==============================================
            # 🔥 分支 1：POSE 模型（姿态 + 人脸识别）
            # ==============================================
            if task_type == "pose":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. 人脸识别
                face_ids = {}
                if frame_count % 10 == 0:
                    face_locs = face_recognition.face_locations(rgb, model="hog")
                    face_encs = face_recognition.face_encodings(rgb, face_locs)
                    
                    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                        sid = "unknown"
                        if known_encodings:
                            dists = face_recognition.face_distance(known_encodings, enc)
                            if len(dists) > 0 and dists.min() < 0.5:
                                sid = known_ids[np.argmin(dists)]
                        face_ids[(left, top, right, bottom)] = sid
                        
                        if sid != "unknown":
                            face_id_cache.add(sid)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, sid, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 2. YOLO-Pose 推理
                pose_results = model(frame, conf=POSE_CONF)
                for res in pose_results:
                    for box, kp in zip(res.boxes.xyxy, res.keypoints):
                        keypoints = kp.data.cpu().numpy().squeeze()
                        behavior = get_behavior(keypoints)
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        # 匹配人脸ID
                        student_code = "unknown"
                        for (fl, ft, fr, fb), sid in face_ids.items():
                            cx2 = (fl + fr) / 2
                            cy2 = (ft + fb) / 2
                            if abs(cx - cx2) < 80 and abs(cy - cy2) < 80:
                                student_code = sid
                                break
                        
                        if student_code != "unknown":
                            if student_code not in student_behaviors:
                                student_behaviors[student_code] = {cls: 0 for cls in CLASS_NAMES_CN}
                            # 统计
                            try:
                                idx = CLASS_NAMES_EN.index(behavior)
                                cn_lbl = CLASS_NAMES_CN[idx]
                                student_behaviors[student_code][cn_lbl] += 1
                            except:
                                pass
                        
                        # 全局统计
                        try:
                            idx = CLASS_NAMES_EN.index(behavior)
                            cn_lbl = CLASS_NAMES_CN[idx]
                            total_count[cn_lbl] += 1
                        except:
                            cn_lbl = behavior

                        # 绘制
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{student_code}:{behavior}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # 记录日志
                        behavior_data.append([
                            frame_count, student_code, behavior, 1.0, int(cx), int(cy),
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])

            # ==============================================
            # 🔥 分支 2：普通 YOLO 检测（你原来的逻辑）
            # ==============================================
            else:
                results = model.predict(frame, verbose=False, conf=0.25)
                det = results[0].boxes.data.cpu().numpy()
                curr_distract_count = 0

                if len(det) > 0:
                    online_targets = tracker.update(det[:, :5], [height, width], [height, width])
                    for t in online_targets:
                        tlbr = t.tlbr
                        tid = t.track_id
                        x1, y1, x2, y2 = map(int, tlbr)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        best_cls = 0
                        min_dist = 1e9
                        for d in det:
                            dcx = (d[0] + d[2]) / 2
                            dcy = (d[1] + d[3]) / 2
                            dist = (cx - dcx) ** 2 + (cy - dcy) ** 2
                            if dist < min_dist:
                                min_dist = dist
                                best_cls = int(d[5])

                        curr_label_en = results[0].names[best_cls]
                        if curr_label_en in DISTRACT_BEHAVIOR:
                            curr_distract_count += 1

                        if 0 <= best_cls < len(CLASS_NAMES_CN):
                            cn_name = CLASS_NAMES_CN[best_cls]
                            total_count[cn_name] += 1

                            if str(tid) not in student_behaviors:
                                student_behaviors[str(tid)] = {c: 0 for c in CLASS_NAMES_CN}
                            student_behaviors[str(tid)][cn_name] += 1

                        label = results[0].names[best_cls]
                        if tid not in track_history:
                            track_history[tid] = deque(maxlen=15)
                        track_history[tid].append(label)
                        final_label = max(set(track_history[tid]), key=track_history[tid].count)

                        color = get_color(best_cls)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID{tid} {final_label}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        behavior_data.append([
                            frame_count, tid, final_label,
                            round(float(t.score), 2), int(cx), int(cy),
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])

                    global last_capture_frame
                    if (frame_count - last_capture_frame >= KEY_FRAME_INTERVAL and
                            curr_distract_count >= GLOBAL_DISTRACT_NUM):
                        key_frame_name = f"global_frame_{frame_count}_distract_{curr_distract_count}.jpg"
                        key_frame_path = os.path.join(KEY_FRAME_SAVE_DIR, key_frame_name)
                        cv2.imwrite(key_frame_path, frame)
                        last_capture_frame = frame_count

            out.write(frame)

        cap.release()
        out.release()

        # 保存 CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'student_code', 'behavior_label', 'confidence', 'cx', 'cy', 'timestamp'])
            writer.writerows(behavior_data)

        # 保存 JSON
        statistics = {
            "total_frames": frame_count,
            "behavior_counts": total_count,
            "student_behaviors": student_behaviors,
            "face_ids": list(face_id_cache),  # 👈 加这个
            "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": current_model_name
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        # 上传 MINIO
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"teacher_{teacher_code}/class_{class_code}/{time_str}_{lesson_section}"
        video_obj = f"{base}/output.mp4"
        json_obj = f"{base}/stats.json"
        csv_obj = f"{base}/tracks.csv"

        key_frame_minio_path = None
        import glob
        key_frames = glob.glob(os.path.join(KEY_FRAME_SAVE_DIR, "global_frame_*.jpg"))
        if key_frames:
            key_frames.sort(reverse=True)
            key_frame_local_path = key_frames[0]
            key_frame_minio_path = f"{base}/keyframe.jpg"
            try:
                minio_client.fput_object(BUCKET_NAME, key_frame_minio_path, key_frame_local_path)
            except:
                pass

        # 上传文件
        try:
            minio_client.fput_object(BUCKET_NAME, video_obj, output_path)
            minio_client.fput_object(BUCKET_NAME, json_obj, json_path)
            minio_client.fput_object(BUCKET_NAME, csv_obj, csv_path)
        except Exception as e:
            print("MINIO ERROR:", e)

        # 写入数据库
        try:
            cursor = db.cursor()
            report_code = f"R{datetime.now().strftime('%Y%m%d%H%M%S')}"
            cursor.execute("""
                INSERT INTO course_reports
                (report_code, teacher_code, class_code, lesson_section, 
                minio_video_path, minio_json_path, minio_csv_path, minio_keyframe_path)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                report_code, teacher_code, class_code, lesson_section,
                video_obj, json_obj, csv_obj, key_frame_minio_path
            ))
            db.commit()
            cursor.close()
        except Exception as e:
            print("DB ERROR:", e)

        return jsonify({
            "status": "success",
            "model": current_model_name,
            "statistics": statistics
        })

    except Exception as e:
        print("FINAL ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/face/by_student", methods=["GET"])
def get_face_by_student():
    student_code = request.args.get("student_code")
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT face_id FROM face_student_mapping WHERE student_code=%s", (student_code,))
        row = cursor.fetchone()
        cursor.close()
        db_tmp.close()
        return jsonify({"face_id": row["face_id"] if row else None})
    except:
        return jsonify({"face_id": None})

@app.route("/api/report/face_ids", methods=["GET"])
def get_report_face_ids():
    report_id = request.args.get("id")
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT minio_json_path FROM course_reports WHERE id=%s", (report_id,))
        report = cursor.fetchone()
        cursor.close()
        db_tmp.close()

        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.loads(data.data)
        face_ids = stats.get("face_ids", [])
        return jsonify({"face_ids": face_ids})
    except:
        return jsonify({"face_ids": []})

# 获取本节课所有学生的 track_id
@app.route("/api/report/students", methods=["GET"])
def get_report_students():
    report_id = request.args.get("id")
    try:
        db_tmp = pymysql.connect(host="localhost", user="root", password="password123", database="user_center_db", charset='utf8mb4')
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT minio_json_path FROM course_reports WHERE id=%s", (report_id,))
        report = cursor.fetchone()
        cursor.close()
        db_tmp.close()

        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.loads(data.data)
        student_ids = list(stats.get("student_behaviors", {}).keys())
        return jsonify({"student_ids": student_ids})
    except:
        return jsonify({"student_ids": []})
    
# 保存 track_id 和 student_code 的映射关系
@app.route("/api/report/bind_student", methods=["POST"])
def bind_student():
    report_id = request.json.get("report_id")
    track_id = request.json.get("track_id")
    student_name = request.json.get("student_name")

    db_tmp = pymysql.connect(host="localhost", user="root", password="password123", database="user_center_db", charset='utf8mb4')
    cursor = db_tmp.cursor()
    cursor.execute("INSERT INTO report_student_mapping (report_id, track_id, student_name) VALUES (%s,%s,%s)",
        (report_id, track_id, student_name))
    db_tmp.commit()
    cursor.close()
    db_tmp.close()
    return jsonify({"status": "ok"})

@app.route("/api/teacher/reports", methods=["GET"])
def teacher_reports():
    teacher_code = request.args.get("teacher_code")
    if not teacher_code:
        return jsonify([])

    # 🔥 每次请求都 NEW 一个数据库连接！！！
    db_temp = pymysql.connect(
        host="localhost",
        user="root",
        password="password123",
        database="user_center_db",
        charset='utf8mb4'
    )

    cursor = db_temp.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT cr.*, c.class_name 
        FROM course_reports cr
        JOIN classes c ON cr.class_code = c.class_code
        WHERE cr.teacher_code = %s
        ORDER BY cr.created_at DESC
    """, (teacher_code,))
    
    data = cursor.fetchall()

    # 🔥 用完立即关闭！
    cursor.close()
    db_temp.close()

    # 禁用缓存
    response = jsonify(data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ========================
# ✅ 获取单节课详细分析
# ========================
@app.route("/api/report/detail", methods=["GET"])
def report_detail():
    try:
        report_id = request.args.get("id")

        db_tmp = pymysql.connect(
            host="localhost", user="root", password="password123", database="user_center_db", charset='utf8mb4'
        )
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        
        cursor.execute("""
            SELECT cr.*, c.class_name 
            FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code
            WHERE cr.id=%s
        """, (report_id,))
        
        report = cursor.fetchone()
        cursor.close()
        db_tmp.close()

        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.loads(data.data)
        
        ai_text = ""
        try:
            ai_path = report['minio_json_path'].replace("stats.json", "ai_report.md")
            ai_obj = minio_client.get_object(BUCKET_NAME, ai_path)
            ai_text = ai_obj.read().decode("utf-8")
        except:
            ai_text = ""

        # ✅ 只返回基础数据，不生成AI
        return jsonify({
            "report": report,
            "statistics": stats,
            "ai_analysis": ai_text
        })

    except Exception as e:
        print("DETAIL ERROR:", e)
        return jsonify({
            "report": {},
            "statistics": {},
            "ai_analysis": ""
        }), 500

@app.route("/api/generate_and_save_ai", methods=["POST"])
def generate_and_save_ai():
    try:
        report_id = request.json.get("id")

        # 1. 查报告（包含关键帧路径 + teacher_code）
        db_tmp = pymysql.connect(host="localhost", user="root", password="password123", database="user_center_db", charset='utf8mb4')
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT cr.*, c.class_name, cr.teacher_code FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code WHERE cr.id=%s
        """, (report_id,))
        report = cursor.fetchone()
        cursor.close()
        db_tmp.close()

        if not report:
            return jsonify({"error": "报告不存在"}), 404

        # 2. 读行为统计
        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.load(data)

        # 3. 关键帧
        key_frame_path = report.get("minio_keyframe_path", "")

        # 4. ✅ 生成 AI（传入 teacher_code！！！）
        from ai_agent import analyze_class_report
        ai_text = analyze_class_report(
            behavior_data=stats["behavior_counts"],
            class_info={
                "class_name": report["class_name"],
                "lesson_section": report["lesson_section"]
            },
            teacher_code=report["teacher_code"],  # ✅ 这里加上
            course_name=report.get("course_name", "课堂行为分析"),
            frame_path=key_frame_path
        )

        # 5. 保存
        ai_path = report['minio_json_path'].replace("stats.json", "ai_report.md")
        minio_client.put_object(
            BUCKET_NAME,
            ai_path,
            io.BytesIO(ai_text.encode("utf-8")),
            length=len(ai_text.encode("utf-8")),
            content_type="text/markdown"
        )

        return jsonify({"ai_analysis": ai_text})

    except Exception as e:
        print("AI SAVE ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/list_videos', methods=['GET'])
def list_videos():
    try:
        objects = minio_client.list_objects(BUCKET_NAME, prefix='processed/', recursive=True)
        video_list = []
        for obj in objects:
            url = minio_client.get_presigned_url("GET", BUCKET_NAME, obj.object_name)
            video_list.append({
                "name": obj.object_name.replace("processed/", ""),
                "url": url,
                "time": obj.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            })
        video_list.sort(key=lambda x: x['time'], reverse=True)
        return jsonify(video_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_history_stat/<filename>', methods=['GET'])
def get_history_stat(filename):
    try:
        stat_object = f"statistics/{filename}"
        local_path = os.path.join(tempfile.gettempdir(), filename)
        minio_client.fget_object(BUCKET_NAME, stat_object, local_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route('/get_video_url', methods=['GET'])
def get_video_url():
    path = request.args.get('path')
    url = minio_client.get_presigned_url("GET", BUCKET_NAME, path)
    return jsonify(url)

# 聊天接口
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    question = data.get("question", "")
    teacher_code = data.get("teacher_code", "")
    session_id = data.get("session_id", "")
    result = chat_agent_api(question, teacher_code, session_id)
    return jsonify(result)  # 直接返回 { answer, thinking_process }

# 获取会话列表
@app.route('/api/chat/sessions', methods=['POST'])
def api_chat_sessions():
    data = request.json
    teacher_code = data.get("teacher_code", "")
    sessions = get_teacher_sessions(teacher_code)
    return jsonify({"sessions": sessions})

@app.route('/api/chat/messages', methods=['POST'])
def api_chat_messages():
    data = request.json
    teacher_code = data.get("teacher_code", "")
    session_id = data.get("session_id", "")
    messages = get_session_messages(teacher_code, session_id)
    return jsonify({"messages": messages})

# 删除单条聊天会话
@app.route("/api/chat/delete_session", methods=["POST"])
def delete_session():
    try:
        data = request.get_json()
        teacher_code = data.get("teacher_code")
        session_id = data.get("session_id")

        if not teacher_code or not session_id:
            return jsonify({"code":400, "msg":"参数缺失"}),400

        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        cursor.execute("""
            DELETE FROM chat_sessions
            WHERE teacher_code=%s AND session_id=%s
        """, (teacher_code, session_id))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"code":200, "msg":"删除成功"})
    except Exception as e:
        print("删除会话错误：",e)
        return jsonify({"code":500, "msg":"删除失败"}),500

# 删除课堂分析报告（含数据库 + MinIO 文件）
@app.route("/api/report/delete", methods=["POST"])
def delete_report():
    try:
        report_id = request.json.get("report_id")
        if not report_id:
            return jsonify({"error": "缺少 report_id"}), 400

        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM course_reports WHERE id=%s", (report_id,))
        report = cursor.fetchone()

        if not report:
            return jsonify({"error": "报告不存在"}), 404

        # 1. 删除 MinIO 文件
        try:
            minio_client.remove_object(BUCKET_NAME, report["minio_video_path"])
            minio_client.remove_object(BUCKET_NAME, report["minio_json_path"])
            if report.get("minio_keyframe_path"):
                minio_client.remove_object(BUCKET_NAME, report["minio_keyframe_path"])
        except:
            pass

        # 2. 删除数据库记录
        cursor.execute("DELETE FROM course_reports WHERE id=%s", (report_id,))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"msg": "删除成功"})

    except Exception as e:
        print("删除报告错误：", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/teacher/course_schedule', methods=['POST'])
def api_course_schedule():
    data = request.json
    teacher_code = data.get("teacher_code", "")
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT week_day, section, class_name, course_name, classroom 
            FROM teacher_course_schedule 
            WHERE teacher_code=%s ORDER BY week_day, section
        """, (teacher_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return jsonify({"list": rows})
    except:
        return jsonify({"list": []})

@app.route("/api/face/mapping", methods=["GET"])
def get_face_mapping():
    class_code = request.args.get("class_code")
    if not class_code:
        return jsonify({"map": {}})
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT face_id, s.student_code, fsm.student_name
            FROM face_student_mapping fsm
            JOIN students s ON fsm.student_code = s.student_code
            WHERE fsm.class_code=%s
        """, (class_code,))
        rows = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        # 转成 {face_id: {id:..., name:...}}
        mapping = {r["face_id"]: {"id": r["student_code"], "name": r["student_name"]} for r in rows}
        return jsonify({"map": mapping})
    except:
        return jsonify({"map": {}})

# 接口1：绑定 face_id → 学生姓名
@app.route("/api/face/bind", methods=["POST"])
def api_face_bind():
    data = request.json
    face_id = data.get("face_id")
    student_code = data.get("student_code")  # 👈 学生ID
    student_name = data.get("student_name")
    class_code = data.get("class_code", None)

    if not face_id or not student_name:
        return jsonify({"code": 400, "msg": "参数错误"}), 400

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor()
        cursor.execute("""
            INSERT INTO face_student_mapping
            (face_id, student_code, student_name, class_code)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                student_code = %s,
                student_name = %s,
                class_code = %s
        """, (
            face_id, student_code, student_name, class_code,
            student_code, student_name, class_code
        ))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()
        return jsonify({"code": 200, "msg": "绑定成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500

# 接口2：获取某个班级所有已绑定的人脸列表
@app.route("/api/face/list", methods=["GET"])
def api_face_list():
    class_code = request.args.get("class_code", None)
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        sql = "SELECT face_id, student_name, class_code FROM face_student_mapping"
        if class_code:
            sql += " WHERE class_code=%s"
            cursor.execute(sql, (class_code,))
        else:
            cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": rows})
    except:
        return jsonify({"list": []})

@app.route("/api/report/save", methods=["POST"])
def save_report():
    d = request.json
    db_tmp = pymysql.connect(**DB_CONFIG)
    cursor = db_tmp.cursor()
    cursor.execute("""
        INSERT INTO student_reports
        (student_code, class_code, lesson_time, normal_posture, raised_hand, looking_down, focus_rate, ai_comment, teacher_score, teacher_comment)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        d["student_code"],  # 👈 改这里
        d["class_code"], d["lesson_time"],
        d["normal_posture"], d["raised_hand"], d["looking_down"], d["focus_rate"],
        d["ai_comment"], d["teacher_score"], d["teacher_comment"]
    ))
    db_tmp.commit()
    cursor.close()
    db_tmp.close()
    return jsonify({"msg": "保存成功"})

@app.route("/api/report/history", methods=["GET"])
def report_history():
    student_code = request.args.get("student_code")
    class_code = request.args.get("class_code")
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT * FROM student_reports
            WHERE student_code=%s AND class_code=%s
            ORDER BY lesson_time DESC
        """, (student_code, class_code))
        lst = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": lst})
    except Exception as e:
        return jsonify({"list": []})

@app.route("/api/student/my-reports", methods=["GET"])
def my_reports():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"list": []})
    
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT * FROM student_reports
            WHERE student_code=%s
            ORDER BY lesson_time DESC
        """, (student_code,))
        lst = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": lst})
    except Exception as e:
        return jsonify({"list": []})

# ========================
# ✅ 获取【单个学生】课堂行为报告（你未来万能接口）
# ========================
@app.route("/api/student/behavior", methods=["GET"])
def get_student_behavior():
    class_code = request.args.get("class_code")
    face_id = request.args.get("face_id")

    if not class_code or not face_id:
        return jsonify({"error": "缺少参数"}), 400

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        # 查这个班级的所有课堂报告
        cursor.execute("""
            SELECT minio_json_path FROM course_reports
            WHERE class_code=%s ORDER BY created_at DESC
        """, (class_code,))
        reports = cursor.fetchall()
        cursor.close()
        db_tmp.close()

        # 汇总该学生所有行为
        total_behaviors = {}
        for r in reports:
            try:
                data = minio_client.get_object(BUCKET_NAME, r["minio_json_path"])
                stats = json.load(data)
                sb = stats.get("student_behaviors", {})
                if face_id in sb:
                    for b, cnt in sb[face_id].items():
                        total_behaviors[b] = total_behaviors.get(b, 0) + cnt
            except:
                continue

        return jsonify({
            "face_id": face_id,
            "behaviors": total_behaviors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# 获取所有班级列表（补全缺失接口）
# ========================
@app.route("/api/class/list", methods=["GET"])
def get_class_list():
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        # 👇 这里加上 class_code！！！
        cursor.execute("SELECT id, class_code, class_name FROM classes")
        class_list = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": class_list})
    except Exception as e:
        return jsonify({"list": []}), 500

# 获取班级学生列表（绑定用）
@app.route("/api/class/students", methods=["GET"])
def get_class_students():
    class_code = request.args.get("class_code")
    if not class_code:
        return jsonify({"students": []})

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT s.student_code, u.name
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            WHERE s.class_code = %s
        """, (class_code,))
        students = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"students": students})
    except:
        return jsonify({"students": []})

# 接口3：批量导入绑定（CSV文件）
@app.route("/api/face/batch_import", methods=["POST"])
def api_face_batch_import():
    if 'file' not in request.files:
        return jsonify({"code": 400, "msg": "请上传文件"}), 400

    file = request.files['file']
    class_code = request.form.get("class_code", None)
    try:
        import csv
        reader = csv.DictReader(file.read().decode("utf-8").splitlines())
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor()
        for row in reader:
            face_id = row.get("face_id")
            student_name = row.get("student_name")
            if face_id and student_name:
                cursor.execute("""
                    INSERT INTO face_student_mapping (face_id, student_name, class_code)
                    VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE student_name=%s
                """, (face_id, student_name, class_code, student_name))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()
        return jsonify({"code": 200, "msg": "批量导入成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500

# 接口4：解除绑定
@app.route("/api/face/unbind", methods=["POST"])
def api_face_unbind():
    face_id = request.json.get("face_id")
    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor()
        cursor.execute("DELETE FROM face_student_mapping WHERE face_id=%s", (face_id,))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()
        return jsonify({"code": 200, "msg": "解除成功"})
    except:
        return jsonify({"code": 500, "msg": "失败"}), 500

@app.route("/api/ai/analyze", methods=["POST"])
def ai_analyze():
    data = request.json
    student_code = data.get("student_code")
    normal = data.get("normal_posture")
    raised = data.get("raised_hand")
    down = data.get("looking_down")
    focus = data.get("focus_rate")

    prompt = f"""
你是小学/中学课堂行为分析师，请用温和、鼓励、专业的语气写一段评语。
行为数据：
正常坐姿：{normal}
举手次数：{raised}
低头次数：{down}
专注度：{focus}%

要求：80字左右，适合家长阅读。
"""

    # 后端调用 Qwen
    client = OpenAI(
        api_key="sk-06abd7a7eb514b3ebd611412f0dc3531",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    res = completion.choices[0].message.content
    return jsonify({"comment": res})

@app.route("/api/report/list", methods=["GET"])
def report_list():
    student_code = request.args.get("student_code")
    class_code = request.args.get("class_code")
    db_tmp = pymysql.connect(**DB_CONFIG)
    cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT * FROM student_reports
        WHERE student_code=%s AND class_code=%s
        ORDER BY lesson_time DESC
    """, (student_code, class_code))
    rows = cursor.fetchall()
    cursor.close()
    db_tmp.close()
    return jsonify({"list": rows})

# 1. 获取学生个人统计数据（今日/本周/本月/学期）
@app.route("/api/student/stats", methods=["GET"])
def student_stats():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"error": "缺少student_code"}), 400

    today = date.today()
    first_day_of_month = today.replace(day=1)

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        # 今日
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS focus,
                   IFNULL(SUM(normal_posture),0) AS lookUp,
                   IFNULL(SUM(looking_down),0) AS disturb
            FROM student_reports
            WHERE student_code=%s AND DATE(lesson_time)=CURDATE()
        """, (student_code,))
        day = cursor.fetchone()

        # 本周
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg
            FROM student_reports
            WHERE student_code=%s AND YEARWEEK(lesson_time)=YEARWEEK(NOW())
        """, (student_code,))
        week = cursor.fetchone()

        # 本月
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg,
                   COUNT(*) AS classCount
            FROM student_reports
            WHERE student_code=%s AND DATE(lesson_time)>=%s
        """, (student_code, first_day_of_month))
        month = cursor.fetchone()

        # 学期平均
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg FROM student_reports
            WHERE student_code=%s
        """, (student_code,))
        semester = cursor.fetchone()

        cursor.close()
        db_tmp.close()

        def round0(v):
            return round(float(v or 0))

        return jsonify({
            "day": {
                "focus": round0(day['focus']),
                "lookUp": round0(day['lookUp']),
                "disturb": round0(day['disturb'])
            },
            "week": {
                "avg": round0(week['avg']),
                "up": 5,
                "bestDay": "周四"
            },
            "month": {
                "avg": round0(month['avg']),
                "progress": 7,
                "classCount": month['classCount']
            },
            "semester": {
                "avg": round0(semester['avg']),
                "level": "A · 优秀" if round0(semester['avg'])>=85 else "B · 良好"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/student/home", methods=["GET"])
def student_home():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"student_name":"","class_name":""})

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT u.name AS student_name, c.class_name
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            JOIN classes c ON s.class_code = c.class_code
            WHERE s.student_code=%s
        """, (student_code,))
        info = cursor.fetchone()
        cursor.close()
        db_tmp.close()
        return jsonify(info)
    except Exception as e:
        return jsonify({"student_name":"","class_name":""})

# ==============================
# 家长端 - 首页数据（孩子信息 + 概况）
# ==============================
@app.route("/api/parent/home", methods=["GET"])
def parent_home():
    user_code = request.args.get("user_code")
    if not user_code:
        return jsonify({})

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        # 1. 获取家长绑定的孩子
        cursor.execute("""
            SELECT p.student_code
            FROM parents p
            WHERE p.user_code=%s
        """, (user_code,))
        parent = cursor.fetchone()
        if not parent:
            return jsonify({})

        student_code = parent["student_code"]

        # 2. 获取孩子信息
        cursor.execute("""
            SELECT u.name AS student_name, c.class_name, s.class_code
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            JOIN classes c ON s.class_code = c.class_code
            WHERE s.student_code=%s
        """, (student_code,))
        student = cursor.fetchone()

        # 3. 今日专注度
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate), 0) AS today_focus
            FROM student_reports
            WHERE student_code=%s AND DATE(lesson_time)=CURDATE()
        """, (student_code,))
        today = cursor.fetchone()

        # 4. 总报告数量
        cursor.execute("""
            SELECT COUNT(*) AS total FROM student_reports
            WHERE student_code=%s
        """, (student_code,))
        total = cursor.fetchone()

        cursor.close()
        db_tmp.close()

        return jsonify({
            "student_name": student["student_name"],
            "class_name": student["class_name"],
            "today_focus": round(float(today["today_focus"])),
            "total_reports": total["total"],
            "student_code": student_code
        })

    except Exception as e:
        print("家长首页错误：", e)
        return jsonify({})

# 2. 班级专注度排行
@app.route("/api/class/rank", methods=["GET"])
def class_rank():
    class_code = request.args.get("class_code")
    if not class_code:
        return jsonify({"rank": []})

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT 
                u.name AS name,
                IFNULL(AVG(r.focus_rate), 0) AS score
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            LEFT JOIN student_reports r 
                ON s.student_code = r.student_code 
                AND r.class_code = %s
            WHERE s.class_code = %s
            GROUP BY s.student_code, u.name
            HAVING score > 0
            ORDER BY score DESC
            LIMIT 10
        """, (class_code, class_code))
        ranks = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"rank": ranks})
    except Exception as e:
        print("排行错误：", e)
        return jsonify({"rank": []})

# 3. 获取学生基本信息（✅ 修复版，多返回 class_code）
@app.route("/api/student/info", methods=["GET"])
def student_info():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"error": "缺少student_code"}), 400

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            # ✅ 这里必须加上 u.name AS student_name
            SELECT u.name AS student_name, s.class_code, c.class_name
            FROM students s
            JOIN users u ON s.user_code = u.user_code   # 关联用户表拿真实姓名
            JOIN classes c ON s.class_code = c.class_code
            WHERE s.student_code=%s
        """, (student_code,))
        info = cursor.fetchone()
        cursor.close()
        db_tmp.close()
        return jsonify(info)
    except Exception as e:
        return jsonify({
            "student_name": "",
            "class_code": "",
            "class_name": ""
        })

# ==============================
# 家校共育 · AI 综合分析建议
# ==============================
@app.route("/api/ai/advice", methods=["GET"])
def ai_advice():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"summary":"","advice":""})

    try:
        db_tmp = pymysql.connect(**DB_CONFIG)
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        # 获取该学生所有报告
        cursor.execute("""
            SELECT focus_rate, normal_posture, raised_hand, looking_down, ai_comment
            FROM student_reports
            WHERE student_code=%s
            ORDER BY lesson_time DESC
        """, (student_code,))
        reports = cursor.fetchall()

        if not reports:
            return jsonify({
                "summary": "暂无历史课堂数据，无法生成分析",
                "advice": "请等待课堂数据生成后再查看"
            })

        # 统计
        total = len(reports)
        avg_focus = round(sum(r["focus_rate"] for r in reports) / total)
        good_posture = sum(r["normal_posture"] for r in reports)
        total_hand = sum(r["raised_hand"] for r in reports)
        total_down = sum(r["looking_down"] for r in reports)

        # AI 总结
        summary = f"""
该生近 {total} 节课平均专注度 {avg_focus}%，整体课堂状态良好。
累计坐姿达标 {good_posture} 次，主动举手发言 {total_hand} 次，分心低头 {total_down} 次。
        """.strip()

        # AI 建议
        advice = f"""
【家校共育建议】
1. 该生专注度表现{ "优秀" if avg_focus>=90 else "良好" if avg_focus>=80 else "一般" }，建议继续保持专注习惯。
2. 主动发言积极性{ "很高" if total_hand>=total*3 else "一般" }，建议多鼓励课堂参与。
3. 分心情况{ "较少" if total_down<=total*1 else "偏多" }，家校共同引导注意力管理。
4. 家庭配合：规律作息、减少电子产品干扰，与学校同步培养学习习惯。
        """.strip()

        cursor.close()
        db_tmp.close()

        return jsonify({
            "summary": summary,
            "advice": advice
        })

    except Exception as e:
        print("AI分析错误:", e)
        return jsonify({"summary":"","advice":""})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)