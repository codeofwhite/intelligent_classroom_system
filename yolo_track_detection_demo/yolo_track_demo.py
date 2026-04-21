from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 或者用 yolov8n-pose.pt 直接带骨架

# 运行追踪
# track 模式会自动调用 ByteTrack
results = model.track(source="demo/Video Project 5.mp4", conf=0.3, iou=0.5, tracker="bytetrack.yaml", show=True)

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # 获取框坐标
    ids = r.boxes.id.cpu().numpy()      # 获取每个人的唯一 ID
    
    for box, track_id in zip(boxes, ids):
        # 1. 这里你可以根据 ID 缓存每个人的动作
        # 2. 比如对这个 box 进行后续的行为识别
        print(f"学生 ID: {int(track_id)} 坐标: {box}")