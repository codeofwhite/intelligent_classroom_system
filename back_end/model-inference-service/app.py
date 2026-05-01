import os
import cv2
import uuid
import json
import csv
import sys
import time
import tempfile
import pymysql
from collections import deque
from ai_agent import analyze_class_report
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from minio import Minio
from chat_agent import chat_agent_api, get_session_messages, get_teacher_sessions
from datetime import datetime
import io

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
app.config['DEBUG'] = True
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
    class_id = data.get("class_id", 1)
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
    class_id = data.get("class_id", 1)
    lesson_section = data.get("lesson_section", "实时课堂")

    # 自动上传逻辑（和你 upload 接口一模一样）
    try:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"teacher_{teacher_code}/class_{class_id}/{time_str}_{lesson_section}"
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
            (report_code, teacher_code, class_id, lesson_section, minio_video_path, minio_json_path, minio_csv_path)
            VALUES (%s,%s,%s,%s,%s,%s,'')
        """, (report_code, teacher_code, int(class_id), lesson_section, video_obj, json_obj))
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

    teacher_code = request.form.get('teacher_code')
    class_id = request.form.get('class_id')
    lesson_section = request.form.get('lesson_section')

    file = request.files['video']
    CLASS_NAMES_EN = ["Raising-Hand", "Reading", "Writing", "Useing-Phone", "Head-down", "Sleep"]
    CLASS_NAMES_CN = ["举手", "看书", "写字", "使用手机", "低头做其他事情", "睡觉"]

    total_count = {cls: 0 for cls in CLASS_NAMES_CN}
    
    # 按照学生的 id 统计
    student_behaviors = {}
    
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

            # ✅ 修复编码器
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用更通用的编码器
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

                        curr_label_en = results[0].names[best_cls]
                        if curr_label_en in DISTRACT_BEHAVIOR:
                            curr_distract_count += 1

                        if 0 <= best_cls < len(CLASS_NAMES_CN):
                            cn_name = CLASS_NAMES_CN[best_cls]
                            total_count[CLASS_NAMES_CN[best_cls]] += 1
                            
                            # ======================
                            # 🔥 按学生单独统计
                            # ======================
                            if str(tid) not in student_behaviors:
                                student_behaviors[str(tid)] = {c:0 for c in CLASS_NAMES_CN}
                            student_behaviors[str(tid)][cn_name] += 1

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

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_id','student_id','behavior_label','confidence','cx','cy','timestamp'])
                writer.writerows(behavior_data)

            statistics = {
                "total_frames": frame_count,
                "behavior_counts": total_count,
                "student_behaviors": student_behaviors,  # 🔥 保存单人数据
                "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)

            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"teacher_{teacher_code}/class_{class_id}/{time_str}_{lesson_section}"

            video_obj = f"{base}/output.mp4"
            json_obj = f"{base}/stats.json"
            csv_obj = f"{base}/tracks.csv"

            # ======================
            # 🔥 上传关键帧到 MINIO（我加的）
            # ======================
            key_frame_local_path = None
            key_frame_minio_path = None

            import glob
            key_frames = glob.glob(os.path.join(KEY_FRAME_SAVE_DIR, f"global_frame_*.jpg"))
            if key_frames:
                # 取最新一张关键帧
                key_frames.sort(reverse=True)
                key_frame_local_path = key_frames[0]
                key_frame_minio_path = f"{base}/keyframe.jpg"

                try:
                    minio_client.fput_object(
                        BUCKET_NAME,
                        key_frame_minio_path,
                        key_frame_local_path
                    )
                    print("✅ 关键帧已上传 MinIO")
                except Exception as e:
                    print("❌ 关键帧上传失败", e)

            # ✅ 安全上传视频、JSON、CSV
            try:
                minio_client.fput_object(BUCKET_NAME, video_obj, output_path)
                minio_client.fput_object(BUCKET_NAME, json_obj, json_path)
                minio_client.fput_object(BUCKET_NAME, csv_obj, csv_path)
            except Exception as e:
                print("MINIO UPLOAD ERROR:", e)

            # ✅ 安全写数据库（加入关键帧路径）
            try:
                cursor = db.cursor()
                class_id_int = int(class_id)
                
                report_code = f"R{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                cursor.execute("""
                    INSERT INTO course_reports
                    (report_code, teacher_code, class_id, lesson_section, 
                    minio_video_path, minio_json_path, minio_csv_path, minio_keyframe_path)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    report_code, teacher_code, class_id_int, lesson_section,
                    video_obj, json_obj, csv_obj, key_frame_minio_path
                ))
                
                db.commit()
                cursor.close()
                print("✅ 数据库插入成功 + 关键帧已记录")
            except Exception as e:
                print("❌ DB ERROR:", e)

            return jsonify({
                "status": "success",
                "statistics": statistics
            })
    except Exception as e:
        print("FINAL ERROR:", e)  # 🔥 这里会打印真实错误
        return jsonify({"error": str(e)}), 500

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
    
# 保存 track_id 和 student_id 的映射关系
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
        JOIN classes c ON cr.class_id = c.id
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
            JOIN classes c ON cr.class_id = c.id
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

# ========================
# ✅ 手动触发 AI 分析（点击才生成）
# ========================
@app.route("/api/generate_and_save_ai", methods=["POST"])
def generate_and_save_ai():
    try:
        report_id = request.json.get("id")

        # 1. 查报告（包含关键帧路径）
        db_tmp = pymysql.connect(host="localhost", user="root", password="password123", database="user_center_db", charset='utf8mb4')
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT cr.*, c.class_name FROM course_reports cr
            JOIN classes c ON cr.class_id = c.id WHERE cr.id=%s
        """, (report_id,))
        report = cursor.fetchone()
        cursor.close()
        db_tmp.close()

        if not report:
            return jsonify({"error": "报告不存在"}), 404

        # 2. 读行为统计
        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.load(data)

        # 3. 🔥 关键：使用当前报告自己的关键帧！！！
        key_frame_path = report.get("minio_keyframe_path", "")

        # 4. 生成 AI
        from ai_agent import analyze_class_report
        ai_text = analyze_class_report(
            behavior_data=stats["behavior_counts"],
            class_info={
                "class_name": report["class_name"],
                "lesson_section": report["lesson_section"]
            },
            course_name="课堂行为分析",
            # 🔥 这里改成真实关键帧路径
            frame_path=key_frame_path
        )

        # 5. 保存 AI 分析结果到 MinIO
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)