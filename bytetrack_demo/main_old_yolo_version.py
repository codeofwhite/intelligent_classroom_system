import os
import sys
import cv2
import csv
import torch
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

# 1. 动态引入 YOLOv7 和 ByteTrack 路径
sys.path.append('./yolov7')
sys.path.append('./ByteTrack')

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from yolox.tracker.byte_tracker import BYTETracker

# 解决 PyTorch 2.6+ 加载权重问题
import functools
torch.load = functools.partial(torch.load, weights_only=False)

class ByteTrackArgs:
    track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

def get_color(category_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), 
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]
    return colors[category_id % len(colors)]

def main(video_path, weight_path, output_path):
    # --- 初始化环境 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weight_path, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    tracker = BYTETracker(ByteTrackArgs())
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    track_history = {} # 用于标签平滑
    behavior_data = [] # 存储 CSV 数据
    frame_count = 0

    print(f"开始推理: {video_path} 使用设备: {device}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # --- 2. 预处理 (Letterbox) ---
        img = letterbox(frame, 640, stride=32)[0]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.unsqueeze(0) if img.ndimension() == 3 else img

        # --- 3. 模型推理 ---
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        
        # --- 4. NMS 过滤 ---
        det = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)[0]

        if det is not None and len(det):
            # 坐标还原到原图尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            det_np = det.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

            # --- 5. ByteTrack 追踪 ---
            # 传给 tracker 的必须是 [x1, y1, x2, y2, score]
            online_targets = tracker.update(det_np[:, :5], [height, width], [height, width])

            for t in online_targets:
                tlbr = t.tlbr
                tid = t.track_id
                
                # 计算当前追踪框中心
                cx, cy = (tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2
                
                # 类别匹配：寻找与该追踪框 IOU 最大的检测框类别
                # (简单做法：寻找中心点距离最近的检测框类别)
                dists = np.linalg.norm(det_np[:, :2] - tlbr[:2], axis=1)
                best_cls = int(det_np[np.argmin(dists), 5])
                label_name = names[best_cls]

                # 标签平滑逻辑 (15帧窗口投票)
                if tid not in track_history: track_history[tid] = deque(maxlen=15)
                track_history[tid].append(label_name)
                smooth_label = max(set(track_history[tid]), key=track_history[tid].count)

                # 存入数据
                behavior_data.append([frame_count, tid, smooth_label, round(t.score, 2), int(cx), int(cy)])

                # --- 6. 可视化绘制 ---
                color = get_color(best_cls)
                x1, y1, x2, y2 = map(int, tlbr)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{tid} {smooth_label}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 实时计数显示
        cv2.putText(frame, f"Frame: {frame_count} | Targets: {len(online_targets) if 'online_targets' in locals() else 0}", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        if frame_count % 100 == 0: print(f"已处理 {frame_count} 帧...")

    # --- 7. 保存结果并分析 ---
    cap.release()
    out.release()
    save_and_analyze(behavior_data)

def save_and_analyze(data):
    df = pd.DataFrame(data, columns=['frame_id', 'student_id', 'behavior_label', 'confidence', 'cx', 'cy'])
    csv_path = "classroom_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至 {csv_path}")

    # 简单分析示例：行为占比
    plt.figure(figsize=(10,6))
    df['behavior_label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Classroom Behavior Distribution")
    plt.savefig("behavior_pie.png")
    print("分析图表已生成。")

if __name__ == "__main__":
    # 请根据实际路径修改
    main(video_path="demo/579977249-1-208 - Trim.mp4", weight_path="yolov7_epoch42.pt", output_path="result.mp4")