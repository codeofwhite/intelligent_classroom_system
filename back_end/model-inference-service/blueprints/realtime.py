"""
实时监测与录制 蓝图
- 开始/停止录制
- 获取实时统计数据
- 获取录制状态
- 视频流 SSE
"""
import os
import json
import time
import tempfile
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from flask import Blueprint, Response, request, jsonify

from config import KEY_FRAME_SAVE_DIR, VIDEO_SOURCE
from shared import (
    model, minio_client, BUCKET_NAME, ByteTrackArgs,
    is_recording, real_time_stats, real_time_logs,
    video_writer, output_video_path, db,
    get_db_connection,
)
from utils import get_color

# 用列表包裹可变状态，便于在函数内修改全局值
_state = {
    "is_recording": False,
    "video_writer": None,
    "output_video_path": None,
    "real_time_stats": {
        "hand_up": 0,
        "study_norm": 0,
        "look_down": 0,
        "abnormal": 0,
    },
    "real_time_logs": [],
    "last_capture_frame": 0,
}

realtime_bp = Blueprint("realtime", __name__)


# ==========================
# 开始录制
# ==========================
@realtime_bp.route('/start_record', methods=['POST'])
def start_record():
    if _state["is_recording"]:
        return jsonify({"status": "already recording"})

    data = request.json
    teacher_code = data.get("teacher_code", "T2025001")
    class_code = data.get("class_code", 1)
    lesson_section = data.get("lesson_section", "实时课堂")

    # 初始化视频保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = tempfile.gettempdir()
    _state["output_video_path"] = os.path.join(tmp_dir, f"record_{timestamp}.mp4")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _state["video_writer"] = cv2.VideoWriter(_state["output_video_path"], fourcc, fps, (w, h))
    cap.release()

    # 清空统计
    _state["real_time_stats"] = {"hand_up": 0, "study_norm": 0, "look_down": 0, "abnormal": 0}
    _state["real_time_logs"] = []

    _state["is_recording"] = True
    return jsonify({"status": "started"})


# ==========================
# 停止录制 + 自动上传入库
# ==========================
@realtime_bp.route('/stop_record', methods=['POST'])
def stop_record():
    if not _state["is_recording"]:
        return jsonify({"status": "not recording"})

    _state["is_recording"] = False
    if _state["video_writer"]:
        _state["video_writer"].release()

    data = request.json
    teacher_code = data.get("teacher_code", "T2025001")
    class_code = data.get("class_code", 1)
    lesson_section = data.get("lesson_section", "实时课堂")

    try:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"teacher_{teacher_code}/class_{class_code}/{time_str}_{lesson_section}"
        video_obj = f"{base}/live.mp4"
        json_obj = f"{base}/live_stats.json"

        stats = {
            "total_frames": 0,
            "behavior_counts": {
                "举手": _state["real_time_stats"]["hand_up"],
                "看书": _state["real_time_stats"]["study_norm"],
                "写字": 0,
                "使用手机": _state["real_time_stats"]["abnormal"],
                "低头做其他事情": _state["real_time_stats"]["look_down"],
                "睡觉": 0,
            }
        }
        json_path = os.path.join(tempfile.gettempdir(), "live_stats.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)

        # 上传 MinIO
        minio_client.fput_object(BUCKET_NAME, video_obj, _state["output_video_path"])
        minio_client.fput_object(BUCKET_NAME, json_obj, json_path)

        # 写入数据库
        db_temp = get_db_connection()
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
@realtime_bp.route('/get_realtime_stats', methods=['GET'])
def get_realtime_stats():
    s = _state["real_time_stats"]
    total = s["hand_up"] + s["study_norm"] + s["look_down"] + s["abnormal"]
    focus_rate = 100 * (s["hand_up"] + s["study_norm"]) / total if total > 0 else 0

    return jsonify({
        "stats": s,
        "focus_rate": round(focus_rate, 1),
        "logs": _state["real_time_logs"][-10:]
    })


# ==========================
# 获取录制状态
# ==========================
@realtime_bp.route('/get_record_status', methods=['GET'])
def get_record_status():
    return jsonify({"recording": _state["is_recording"]})


# ========================
# 实时视频流
# ========================
def generate_frames():
    from ultralytics import YOLO
    camera = cv2.VideoCapture(VIDEO_SOURCE)
    from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
    tracker = BYTETracker(ByteTrackArgs())
    track_history = {}

    while True:
        success, frame = camera.read()
        if not success:
            break

        # 保存视频（如果正在录制）
        if _state["is_recording"] and _state["video_writer"]:
            _state["video_writer"].write(frame)

        results = model.predict(frame, imgsz=640, verbose=False, half=True, conf=0.25)
        det = results[0].boxes.data.cpu().numpy()

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

                if final_label == "Raising-Hand":
                    rt_hand += 1
                elif final_label in ("Reading", "Writing"):
                    rt_study_norm += 1
                elif final_label == "Head-down":
                    rt_look_down += 1
                elif final_label in ("Useing-Phone", "Sleep"):
                    rt_abnormal += 1

        # 更新全局实时数据
        if _state["is_recording"]:
            _state["real_time_stats"]["hand_up"] = rt_hand
            _state["real_time_stats"]["study_norm"] = rt_study_norm
            _state["real_time_stats"]["look_down"] = rt_look_down
            _state["real_time_stats"]["abnormal"] = rt_abnormal

            now = time.strftime("%H:%M:%S")
            log = f"[{now}] 举手:{rt_hand} 抬头:{rt_study_norm} 低头:{rt_look_down}"
            _state["real_time_logs"].append(log)

        frame_display = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@realtime_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')