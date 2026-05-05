"""
视频上传分析 蓝图
- 上传视频进行行为分析
- 列出已处理视频
- 获取历史统计
- 获取视频预签名URL
"""
import os
import sys
import csv
import json
import time
import glob
import tempfile
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from config import (
    KEY_FRAME_SAVE_DIR, KEY_FRAME_INTERVAL,
    GLOBAL_DISTRACT_NUM
)
import shared
from shared import minio_client, BUCKET_NAME, ByteTrackArgs, get_db_connection
from utils import get_color

video_bp = Blueprint("video", __name__)


@video_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    teacher_code = request.form.get('teacher_code')
    class_code = request.form.get('class_code')
    lesson_section = request.form.get('lesson_section')
    file = request.files['video']

    face_id_cache = set()

    if shared.current_model_config is None:
        return jsonify({"error": "No model selected"}), 400

    CLASS_NAMES_EN = shared.current_model_config["labels_en"]
    CLASS_NAMES_CN = shared.current_model_config["labels_cn"]
    FOCUS_BEHAVIOR = shared.current_model_config["focus"]
    DISTRACT_BEHAVIOR = shared.current_model_config["distract"]
    task_type = shared.current_model_config["task"]

    total_count = {cls: 0 for cls in CLASS_NAMES_CN}
    student_behaviors = {}
    frame_count = 0
    behavior_data = []

    # 每次请求独立的临时目录，避免跨视频污染
    request_tmp = tempfile.mkdtemp(prefix="upload_")
    request_keyframe_dir = os.path.join(request_tmp, "keyframes")
    os.makedirs(request_keyframe_dir, exist_ok=True)

    input_path = os.path.join(request_tmp, "input.mp4")
    output_path = os.path.join(request_tmp, "output.mp4")
    csv_path = os.path.join(request_tmp, "tracks.csv")
    json_path = os.path.join(request_tmp, "result.json")

    try:
        file.save(input_path)
        time.sleep(0.1)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
        tracker = BYTETracker(ByteTrackArgs())
        track_history = {}
        last_capture_frame = 0
        # 记录所有关键帧信息：[(帧号, 分心人数, 文件路径)]
        keyframe_records = []
        # 固定间隔采样关键帧（不管有没有分心都拍一张，用于"正常课堂"场景）
        sample_interval = max(KEY_FRAME_INTERVAL * 3, 90)  # 至少每90帧采样一张
        last_sample_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ==============================================
            # 分支 1：POSE 模型（姿态 + 人脸识别）
            # ==============================================
            if task_type == "pose":
                import face_recognition
                import pose_utils
                from pose_utils import get_behavior, POSE_CONF

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. 人脸识别
                face_ids = {}
                if frame_count % 10 == 0:
                    face_locs = face_recognition.face_locations(rgb, model="hog")
                    face_encs = face_recognition.face_encodings(rgb, face_locs)

                    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                        sid = "unknown"
                        if pose_utils.known_encodings:
                            dists = face_recognition.face_distance(pose_utils.known_encodings, enc)
                            if len(dists) > 0 and dists.min() < 0.5:
                                sid = pose_utils.known_ids[np.argmin(dists)]
                        face_ids[(left, top, right, bottom)] = sid

                        if sid != "unknown":
                            face_id_cache.add(sid)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, sid, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 2. YOLO-Pose 推理
                pose_results = shared.model(frame, conf=POSE_CONF)
                for res in pose_results:
                    if res.keypoints is None or res.boxes is None:
                        continue
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

                        behavior_data.append([
                            frame_count, student_code, behavior, 1.0, int(cx), int(cy),
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])

            # ==============================================
            # 分支 2：普通 YOLO 检测
            # ==============================================
            else:
                results = shared.model.predict(frame, verbose=False, conf=0.25)
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

                    # 策略1：分心人数达到阈值时拍摄关键帧
                    if (frame_count - last_capture_frame >= KEY_FRAME_INTERVAL and
                            curr_distract_count >= GLOBAL_DISTRACT_NUM):
                        key_frame_name = f"keyframe_{frame_count}_distract_{curr_distract_count}.jpg"
                        key_frame_path = os.path.join(request_keyframe_dir, key_frame_name)
                        cv2.imwrite(key_frame_path, frame)
                        keyframe_records.append((frame_count, curr_distract_count, key_frame_path))
                        last_capture_frame = frame_count

                    # 策略2：定期采样（保证正常课堂也有关键帧）
                    elif frame_count - last_sample_frame >= sample_interval:
                        key_frame_name = f"sample_{frame_count}.jpg"
                        key_frame_path = os.path.join(request_keyframe_dir, key_frame_name)
                        cv2.imwrite(key_frame_path, frame)
                        keyframe_records.append((frame_count, curr_distract_count, key_frame_path))
                        last_sample_frame = frame_count

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
            "face_ids": list(face_id_cache),
            "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": shared.current_model_name
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        # 上传 MINIO
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"teacher_{teacher_code}/class_{class_code}/{time_str}_{lesson_section}"
        video_obj = f"{base}/output.mp4"
        json_obj = f"{base}/stats.json"
        csv_obj = f"{base}/tracks.csv"

        # 选取最佳关键帧：优先选分心人数最多的，其次选视频中间的采样帧
        key_frame_minio_path = None
        key_frame_local_path = None
        if keyframe_records:
            # 按分心人数降序排序，选分心最多的那张
            keyframe_records.sort(key=lambda x: x[1], reverse=True)
            best = keyframe_records[0]
            key_frame_local_path = best[2]
        elif not keyframe_records:
            # 没有任何关键帧时（极端情况），从视频中间截一帧
            mid_frame = frame_count // 2
            cap_seek = cv2.VideoCapture(output_path)
            cap_seek.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, mid_img = cap_seek.read()
            cap_seek.release()
            if ret:
                key_frame_local_path = os.path.join(request_keyframe_dir, "mid_frame.jpg")
                cv2.imwrite(key_frame_local_path, mid_img)

        if key_frame_local_path and os.path.exists(key_frame_local_path):
            key_frame_minio_path = f"{base}/keyframe.jpg"
            try:
                minio_client.fput_object(BUCKET_NAME, key_frame_minio_path, key_frame_local_path)
            except:
                pass

        try:
            minio_client.fput_object(BUCKET_NAME, video_obj, output_path)
            minio_client.fput_object(BUCKET_NAME, json_obj, json_path)
            minio_client.fput_object(BUCKET_NAME, csv_obj, csv_path)
        except Exception as e:
            print("MINIO ERROR:", e)

        # 写入数据库
        try:
            db_tmp = get_db_connection()
            cursor = db_tmp.cursor()
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
            db_tmp.commit()
            cursor.close()
            db_tmp.close()
        except Exception as e:
            print("DB ERROR:", e)

        # 清理临时目录（MinIO 已保存，本地不再需要）
        try:
            import shutil
            shutil.rmtree(request_tmp, ignore_errors=True)
        except:
            pass

        return jsonify({
            "status": "success",
            "model": shared.current_model_name,
            "statistics": statistics
        })

    except Exception as e:
        # 异常时也清理临时目录
        try:
            import shutil
            shutil.rmtree(request_tmp, ignore_errors=True)
        except:
            pass
        print("FINAL ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@video_bp.route('/list_videos', methods=['GET'])
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


@video_bp.route('/get_history_stat/<filename>', methods=['GET'])
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


@video_bp.route('/get_video_url', methods=['GET'])
def get_video_url():
    path = request.args.get('path')
    url = minio_client.get_presigned_url("GET", BUCKET_NAME, path)
    return jsonify(url)