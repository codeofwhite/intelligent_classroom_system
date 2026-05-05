"""
实时监测与录制 蓝图
- 开始/停止录制
- 获取实时统计数据
- 获取录制状态
- 视频流 MJPEG
"""
import os
import csv
import json
import time
import tempfile
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from flask import Blueprint, Response, request, jsonify

from config import (
    KEY_FRAME_SAVE_DIR, KEY_FRAME_INTERVAL,
    GLOBAL_DISTRACT_NUM, VIDEO_SOURCE
)
import shared
from shared import (
    minio_client, BUCKET_NAME, ByteTrackArgs,
    get_db_connection,
)
from utils import get_color

_state = {
    "is_recording": False,
    "video_writer": None,
    "output_video_path": None,
    "request_tmp": None,       # 本次录制的临时目录
    "behavior_data": [],       # [(frame_id, tid, label, conf, cx, cy, timestamp)]
    "student_behaviors": {},   # {tid: {行为: 次数}}
    "total_count": {},         # {行为标签: 总次数}
    "face_id_cache": set(),
    "keyframe_records": [],    # [(frame_count, distract_count, path)]
    "last_capture_frame": 0,
    "last_sample_frame": 0,
    "real_time_stats": {
        "hand_up": 0,
        "study_norm": 0,
        "look_down": 0,
        "abnormal": 0,
    },
    "real_time_logs": [],
    "frame_count": 0,
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
    class_code = data.get("class_code", "C2025001")
    lesson_section = data.get("lesson_section", "实时课堂")

    # 独立临时目录
    request_tmp = tempfile.mkdtemp(prefix="live_")
    _state["request_tmp"] = request_tmp

    # 初始化视频保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(request_tmp, f"record_{timestamp}.mp4")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    cap.release()

    _state["video_writer"] = video_writer
    _state["output_video_path"] = output_video_path

    # 清空所有累积数据
    _state["behavior_data"] = []
    _state["student_behaviors"] = {}
    _state["total_count"] = {}
    _state["face_id_cache"] = set()
    _state["keyframe_records"] = []
    _state["last_capture_frame"] = 0
    _state["last_sample_frame"] = 0
    _state["real_time_stats"] = {"hand_up": 0, "study_norm": 0, "look_down": 0, "abnormal": 0}
    _state["real_time_logs"] = []
    _state["frame_count"] = 0

    _state["is_recording"] = True
    return jsonify({"status": "started"})


# ==========================
# 停止录制 + 保存完整数据 + 上传 MinIO
# ==========================
@realtime_bp.route('/stop_record', methods=['POST'])
def stop_record():
    if not _state["is_recording"]:
        return jsonify({"status": "not recording"})

    _state["is_recording"] = False
    if _state["video_writer"]:
        _state["video_writer"].release()
        _state["video_writer"] = None

    data = request.json
    teacher_code = data.get("teacher_code", "T2025001")
    class_code = data.get("class_code", "C2025001")
    lesson_section = data.get("lesson_section", "实时课堂")

    request_tmp = _state.get("request_tmp", tempfile.gettempdir())

    try:
        # ========================
        # 1. 保存 CSV（行为轨迹）
        # ========================
        csv_path = os.path.join(request_tmp, "tracks.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'track_id', 'behavior_label', 'confidence', 'cx', 'cy', 'timestamp'])
            writer.writerows(_state["behavior_data"])

        # ========================
        # 2. 保存 JSON（完整统计）
        # ========================
        statistics = {
            "total_frames": _state["frame_count"],
            "behavior_counts": _state["total_count"],
            "student_behaviors": _state["student_behaviors"],
            "face_ids": list(_state["face_id_cache"]),
            "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": shared.current_model_name,
            "mode": "realtime_recording"
        }
        json_path = os.path.join(request_tmp, "stats.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        # ========================
        # 3. 选取关键帧
        # ========================
        key_frame_minio_path = None
        key_frame_local_path = None
        if _state["keyframe_records"]:
            _state["keyframe_records"].sort(key=lambda x: x[1], reverse=True)
            key_frame_local_path = _state["keyframe_records"][0][2]
        else:
            # 兜底：从视频中间截一帧
            cap_seek = cv2.VideoCapture(_state["output_video_path"])
            mid = _state["frame_count"] // 2
            cap_seek.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, mid_img = cap_seek.read()
            cap_seek.release()
            if ret:
                key_frame_local_path = os.path.join(request_tmp, "keyframe_mid.jpg")
                cv2.imwrite(key_frame_local_path, mid_img)

        # ========================
        # 4. 上传 MinIO
        # ========================
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"teacher_{teacher_code}/class_{class_code}/{time_str}_{lesson_section}"
        video_obj = f"{base}/live_output.mp4"
        json_obj = f"{base}/live_stats.json"
        csv_obj = f"{base}/live_tracks.csv"

        try:
            minio_client.fput_object(BUCKET_NAME, video_obj, _state["output_video_path"])
            minio_client.fput_object(BUCKET_NAME, json_obj, json_path)
            minio_client.fput_object(BUCKET_NAME, csv_obj, csv_path)
        except Exception as e:
            print("MINIO ERROR:", e)

        if key_frame_local_path and os.path.exists(key_frame_local_path):
            key_frame_minio_path = f"{base}/keyframe.jpg"
            try:
                minio_client.fput_object(BUCKET_NAME, key_frame_minio_path, key_frame_local_path)
            except:
                pass

        # ========================
        # 5. 写入数据库
        # ========================
        try:
            db_temp = get_db_connection()
            cursor = db_temp.cursor()
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
            db_temp.commit()
            cursor.close()
            db_temp.close()
        except Exception as e:
            print("DB ERROR:", e)

        # ========================
        # 6. 清理临时目录
        # ========================
        try:
            import shutil
            shutil.rmtree(request_tmp, ignore_errors=True)
        except:
            pass

        return jsonify({
            "status": "stopped & saved & uploaded",
            "statistics": statistics
        })
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
        "logs": _state["real_time_logs"][-20:]
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

        results = shared.model.predict(frame, imgsz=640, verbose=False, half=True, conf=0.25)
        det = results[0].boxes.data.cpu().numpy()

        rt_hand = 0
        rt_study_norm = 0
        rt_look_down = 0
        rt_abnormal = 0

        if len(det) > 0:
            online_targets = tracker.update(
                det[:, :5],
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]]
            )
            for t in online_targets:
                tlbr = t.tlbr
                tid = t.track_id
                x1, y1, x2, y2 = map(int, tlbr)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                best_cls = 0
                min_dist = 1e9
                for d in det:
                    dcx = (d[0]+d[2])/2
                    dcy = (d[1]+d[3])/2
                    dist = (dcx - cx)**2 + (dcy - cy)**2
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
                cv2.putText(frame, f"ID{tid} {final_label}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 实时统计
                if final_label == "Raising-Hand":
                    rt_hand += 1
                elif final_label in ("Reading", "Writing"):
                    rt_study_norm += 1
                elif final_label == "Head-down":
                    rt_look_down += 1
                elif final_label in ("Useing-Phone", "Sleep"):
                    rt_abnormal += 1

                # ===== 录制中：积累行为数据 =====
                if _state["is_recording"]:
                    _state["frame_count"] += 1
                    frame_count = _state["frame_count"]

                    _state["behavior_data"].append([
                        frame_count, tid, final_label,
                        round(float(t.score), 2), int(cx), int(cy),
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    ])

                    # 累积行为计数
                    _state["total_count"][final_label] = _state["total_count"].get(final_label, 0) + 1

                    # 累积学生行为
                    tid_str = str(tid)
                    if tid_str not in _state["student_behaviors"]:
                        _state["student_behaviors"][tid_str] = {}
                    _state["student_behaviors"][tid_str][final_label] = \
                        _state["student_behaviors"][tid_str].get(final_label, 0) + 1

                    # 关键帧：分心人数达阈值
                    curr_distract = rt_look_down + rt_abnormal
                    if (frame_count - _state["last_capture_frame"] >= KEY_FRAME_INTERVAL and
                            curr_distract >= GLOBAL_DISTRACT_NUM):
                        tmp = _state.get("request_tmp", tempfile.gettempdir())
                        kf_path = os.path.join(tmp, f"keyframe_{frame_count}.jpg")
                        cv2.imwrite(kf_path, frame)
                        _state["keyframe_records"].append((frame_count, curr_distract, kf_path))
                        _state["last_capture_frame"] = frame_count

                    # 关键帧：定期采样
                    sample_interval = max(KEY_FRAME_INTERVAL * 3, 90)
                    if frame_count - _state["last_sample_frame"] >= sample_interval:
                        tmp = _state.get("request_tmp", tempfile.gettempdir())
                        kf_path = os.path.join(tmp, f"sample_{frame_count}.jpg")
                        cv2.imwrite(kf_path, frame)
                        _state["keyframe_records"].append((frame_count, curr_distract, kf_path))
                        _state["last_sample_frame"] = frame_count

        # 更新全局实时数据
        if _state["is_recording"]:
            _state["real_time_stats"]["hand_up"] = rt_hand
            _state["real_time_stats"]["study_norm"] = rt_study_norm
            _state["real_time_stats"]["look_down"] = rt_look_down
            _state["real_time_stats"]["abnormal"] = rt_abnormal

            now = time.strftime("%H:%M:%S")
            log = f"[{now}] 举手:{rt_hand} 正常:{rt_study_norm} 低头:{rt_look_down} 异常:{rt_abnormal}"
            _state["real_time_logs"].append(log)

        frame_display = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@realtime_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')