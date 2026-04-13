import cv2
import csv
import time
import sys
import numpy as np
from collections import deque
from ultralytics import YOLO

# 1. 动态添加 ByteTrack 路径
sys.path.append('./ByteTrack') 
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

class ByteTrackArgs:
    track_thresh = 0.2
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

def get_color(category_id):
    """为 14 种课堂行为生成高区分度颜色 (BGR 格式)"""
    # 按照 14 种行为预设的调色盘
    colors = [
        (255, 0, 0),      # 0: 纯蓝
        (0, 255, 0),      # 1: 纯绿
        (0, 0, 255),      # 2: 纯红
        (255, 255, 0),    # 3: 青色
        (255, 0, 255),    # 4: 品红
        (0, 255, 255),    # 5: 纯黄
        (128, 0, 0),      # 6: 深蓝
        (0, 128, 0),      # 7: 深绿
        (0, 0, 128),      # 8: 深红
        (128, 128, 0),    # 9: 鸭羽绿
        (128, 0, 128),    # 10: 紫色
        (0, 128, 128),    # 11: 橄榄色
        (255, 128, 0),    # 12: 橙色
        (0, 128, 255),    # 13: 天蓝色
    ]
    # 使用取余算法，即使类别超过 14 也不会报错
    return colors[category_id % len(colors)]

def main(video_path, model_path, output_path):
    model = YOLO(model_path, task='detect')
    tracker = BYTETracker(ByteTrackArgs())
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 25
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    track_history = {}
    print("开始推理")

    behavior_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. YOLO 检测 (使用 OpenVINO)
        results = model.predict(frame, device="cpu", conf=0.25, verbose=False)
        det = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        if len(det) > 0:
            # 2. ByteTrack 跟踪
            online_targets = tracker.update(det[:, :5], [height, width], [height, width])

            for t in online_targets:
                tlbr = t.tlbr
                tid = t.track_id
                
                # --- 核心逻辑：找到与当前跟踪框重合度最高的检测类别 ---
                # 计算跟踪框中心点
                tx = (tlbr[0] + tlbr[2]) / 2
                ty = (tlbr[1] + tlbr[3]) / 2
                
                # 在 det 中找距离最近的检测框作为其类别来源
                best_cls = 0
                min_dist = float('inf')
                for d in det:
                    dx = (d[0] + d[2]) / 2
                    dy = (d[1] + d[3]) / 2
                    dist = (tx - dx)**2 + (ty - dy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_cls = int(d[5])

                # 标签与平滑
                label_name = results[0].names[best_cls]
                if tid not in track_history:
                    track_history[tid] = deque(maxlen=15)
                track_history[tid].append(label_name)
                smooth_label = max(set(track_history[tid]), key=track_history[tid].count)

                behavior_data.append([
                    frame_count, 
                    tid, 
                    smooth_label, 
                    round(float(t.score), 2), 
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])

                # --- 3. 绘制 (根据类别选择颜色) ---
                color = get_color(best_cls)
                x1, y1, x2, y2 = map(int, tlbr)
                
                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 画标签背景
                text = f"ID:{tid} {smooth_label}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        out.write(frame)
        
    # --- 【新增】推理结束后，保存为 CSV 文件 ---
    csv_filename = "classroom_analysis_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['frame_id', 'student_id', 'behavior_label', 'confidence', 'timestamp'])
        # 写入数据
        writer.writerows(behavior_data)
        
    print(f"✅ 行为轨迹数据已保存至: {csv_filename}")
        
    cap.release()
    out.release()
    print("\n✅ 处理完成！")

if __name__ == "__main__":
    main("demo/test.mp4", "best_last_openvino_model", "output_xpu_colored.mp4")