import cv2
from ultralytics import YOLO
import numpy as np

# 级联方案需要两个模型：一个负责稳住人，一个负责认姿态
det_model = YOLO('yolov8n_best.pt')        # 你的 CrowdHuman 优化检测模型
pose_model = YOLO('yolov8n-pose.pt')     # 官方轻量级 Pose 模型

# 2. 打开视频源
input_video_path = "demo/testchild.mp4"
cap = cv2.VideoCapture(input_video_path)

# 获取视频属性：宽、高、帧率
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. 创建视频写入器 (保存结果)
# 使用 'mp4v' 编码，保存为 .mp4
output_video_path = "tracking_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"正在通过级联方案处理视频并保存至: {output_video_path} ...")

# 4. 逐帧处理
results_stream = det_model.track(
    source=input_video_path, 
    conf=0.25, 
    iou=0.5, 
    persist=True, 
    tracker="bytetrack.yaml",
    stream=True
)

for r in results_stream:
    # 获取当前帧的原始图像 (OpenCV 格式)
    frame = r.orig_img
    
    # 用于绘制最终结果的帧
    annotated_frame = frame.copy()
    
    if r.boxes.id is not None:
        # 获取所有人的 ID 和 框坐标 (xyxy)
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        
        for box, track_id in zip(boxes, ids):
            # --- 步骤 1: 抠图 (Crop) ---
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue

            # --- 步骤 2: 单人姿态估计 ---
            pose_results = pose_model(person_crop, verbose=False)
            
            # --- 步骤 3: 增加判空检查 (修正报错的关键) ---
            # 只有当 Pose 模型真的在抠图中找到了关键点，才进行后续操作
            if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.data) > 0:
                # 获取第一人的关键点数据
                kpts_rel = pose_results[0].keypoints.data[0].cpu().numpy() 
                
                # 定义骨架连线
                skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], 
                            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], 
                            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

                for limb in skeleton:
                    # 索引减1对齐 (YOLO 关键点是 1-17, 数组是 0-16)
                    idx1, idx2 = limb[0]-1, limb[1]-1
                    
                    # 确保索引不越界且置信度足够
                    if idx1 < len(kpts_rel) and idx2 < len(kpts_rel):
                        kpt1, kpt2 = kpts_rel[idx1], kpts_rel[idx2]
                        if kpt1[2] > 0.5 and kpt2[2] > 0.5:
                            x_p1, y_p1 = int(kpt1[0] + x1), int(kpt1[1] + y1)
                            x_p2, y_p2 = int(kpt2[0] + x1), int(kpt2[1] + y1)
                            cv2.line(annotated_frame, (x_p1, y_p1), (x_p2, y_p2), (0, 255, 255), 2)
                
                # 绘制关键点点位
                for kpt in kpts_rel:
                    if kpt[2] > 0.5:
                        x_p, y_p = int(kpt[0] + x1), int(kpt[1] + y1)
                        cv2.circle(annotated_frame, (x_p, y_p), 3, (0, 0, 255), -1)

            # 无论有没有 Pose 结果，都画出检测框和 ID (保持追踪的视觉反馈)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 绘制追踪结果的自带方法在此处可能会导致覆盖我们手动画的骨架，
    # 所以我们不使用 annotated_frame = r.plot()，而是直接写入我们手动绘制的
    out.write(annotated_frame)

# 5. 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print("处理完成！你可以打开 tracking_result.mp4 观察后排识别效果了。")