import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

# 2. 读取你的教室实战视频或图像
source = 'demo/testchild.mp4'  # 替换为你的实际视频路径
results = model.predict(source, stream=True, conf=0.5)

for r in results:
    img = r.orig_img.copy()
    keypoints = r.keypoints.data  # 关键点数据: [人索引, 17, 3] (x, y, conf)

    for i, kpts in enumerate(keypoints):
        # 提取关键点坐标 (yolo 默认 17 个点)
        # 0:鼻尖, 5:左肩, 6:右肩, 9:左手腕, 10:右手腕
        try:
            left_shoulder_y = kpts[5][1]
            right_shoulder_y = kpts[6][1]
            left_wrist_y = kpts[9][1]
            right_wrist_y = kpts[10][1]

            # --- 动作逻辑初探：举手判定 ---
            # 逻辑：手腕的 Y 坐标小于（即高于）肩膀的 Y 坐标
            is_raising_hand = False
            if (left_wrist_y < left_shoulder_y - 20) or (right_wrist_y < right_shoulder_y - 20):
                is_raising_hand = True

            # 绘制骨架
            # YOLO 的 results.plot() 可以自动画框和点
            # 这里我们手动加个标签区分动作
            label = "Raising Hand" if is_raising_hand else "Normal"
            color = (0, 255, 0) if not is_raising_hand else (0, 0, 255)
            
            # 获取人体的框坐标
            box = r.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        except IndexError:
            continue

    cv2.imshow('Pose Action Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()