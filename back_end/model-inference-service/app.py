import os
import cv2
import uuid
import json
import csv
import sys
import time
import tempfile
from collections import deque
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from minio import Minio
from datetime import datetime

# ==========================
# 全局关键帧配置（全班分心才截图）
# ==========================
FOCUS_BEHAVIOR = ["Raising-Hand", "Reading", "Writing"]
DISTRACT_BEHAVIOR = ["Useing-Phone", "Head-down", "Sleep"]

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
CORS(app)

MODELS_DIR = "models"
available_models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
current_model_name = available_models[0] if available_models else None
model = YOLO(os.path.join(MODELS_DIR, current_model_name), task='detect') if current_model_name else None

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
video_source = "http://192.168.1.196:8080/video"

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

# ========================
# 实时流
# ========================
def generate_frames():
    camera = cv2.VideoCapture(video_source)
    tracker = BYTETracker(ByteTrackArgs())
    track_history = {}

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model.predict(frame, imgsz=640, verbose=False, half=True, conf=0.25)
        det = results[0].boxes.data.cpu().numpy()

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
    global model, current_model_name
    target_model = request.json.get('model_name')
    if target_model not in os.listdir(MODELS_DIR):
        return jsonify({"status": "error", "msg": "模型不存在"}), 404
    try:
        model = YOLO(os.path.join(MODELS_DIR, target_model), task='detect')
        current_model_name = target_model
        return jsonify({"status": "success", "current": current_model_name})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

# ========================
# ✅ 修复完成的上传分析
# ========================
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    unique_filename = f"{uuid.uuid4()}.mp4"

    CLASS_NAMES_EN = ["Raising-Hand", "Reading", "Writing", "Useing-Phone", "Head-down", "Sleep"]
    CLASS_NAMES_CN = ["举手", "看书", "写字", "使用手机", "低头做其他事情", "睡觉"]

    total_count = {cls: 0 for cls in CLASS_NAMES_CN}
    frame_count = 0
    behavior_data = []

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, "input.mp4")
            output_path = os.path.join(tmpdirname, "output.mp4")
            csv_path = os.path.join(tmpdirname, "tracks.csv")
            json_path = os.path.join(tmpdirname, "result.json")
            file.save(input_path)

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # ✅ tracker 只创建一次
            tracker = BYTETracker(ByteTrackArgs())
            track_history = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

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
                            dcx = (d[0]+d[2])/2
                            dcy = (d[1]+d[3])/2
                            dist = (cx - dcx)**2 + (cy - dcy)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_cls = int(d[5])

                        # ✅ 统计分心人数（移到正确位置）
                        curr_label_en = results[0].names[best_cls]
                        if curr_label_en in DISTRACT_BEHAVIOR:
                            curr_distract_count += 1

                        if 0 <= best_cls < len(CLASS_NAMES_CN):
                            total_count[CLASS_NAMES_CN[best_cls]] += 1

                        label = results[0].names[best_cls]
                        if tid not in track_history:
                            track_history[tid] = deque(maxlen=15)
                        track_history[tid].append(label)
                        final_label = max(set(track_history[tid]), key=track_history[tid].count)

                        color = get_color(best_cls)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID{tid} {final_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        behavior_data.append([
                            frame_count, tid, final_label,
                            round(float(t.score), 2), int(cx), int(cy),
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])

                    # ✅ 全局关键帧（正确位置）
                    global last_capture_frame
                    if (frame_count - last_capture_frame >= KEY_FRAME_INTERVAL and
                        curr_distract_count >= GLOBAL_DISTRACT_NUM):
                        key_frame_name = f"global_frame_{frame_count}_distract_{curr_distract_count}.jpg"
                        key_frame_path = os.path.join(KEY_FRAME_SAVE_DIR, key_frame_name)
                        cv2.imwrite(key_frame_path, frame)
                        last_capture_frame = frame_count
                        print(f"✅ 全局关键帧：{key_frame_name}")

                out.write(frame)

            cap.release()
            out.release()

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_id', 'student_id', 'behavior_label', 'confidence', 'cx', 'cy', 'timestamp'])
                writer.writerows(behavior_data)

            statistics = {
                "video_id": unique_filename,
                "total_frames": frame_count,
                "behavior_counts": total_count,
                "class_names": CLASS_NAMES_CN,
                "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:6]
            video_object = f"processed/{timestamp}_video_{uid}.mp4"
            json_object = f"statistics/{timestamp}_stats_{uid}.json"
            csv_object = f"tracks/{timestamp}_track_{uid}.csv"

            minio_client.fput_object(BUCKET_NAME, video_object, output_path, content_type="video/mp4")
            minio_client.fput_object(BUCKET_NAME, json_object, json_path, content_type="application/json")
            minio_client.fput_object(BUCKET_NAME, csv_object, csv_path, content_type="text/csv")

            video_url = minio_client.get_presigned_url("GET", BUCKET_NAME, video_object)

            return jsonify({
                "status": "success",
                "video_url": video_url,
                "statistics": statistics,
                "msg": "分析完成"
            })

    except Exception as e:
        print(f"ERROR: {e}")
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

# ========================
# 家长端接口
# ========================
@app.route('/api/student/report/<int:student_id>', methods=['GET'])
def get_student_report(student_id):
    return {
        "student_id": student_id,
        "student_name": "测试学生",
        "total_frames": 1200,
        "behavior_counts": {
            "举手": 8, "看书": 680, "写字": 240,
            "使用手机": 12, "低头": 36, "睡觉": 4
        },
        "analyzed_time": "2026-04-24 22:00:00"
    }

@app.route('/api/ai_advice', methods=['GET'])
def get_ai_advice():
    return jsonify({
        "summary": "学生课堂专注度良好，存在偶尔低头、使用手机现象，需加强引导。",
        "advice": "1. 控制电子产品使用\n2. 家校共同监督课堂状态\n3. 鼓励积极互动"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)