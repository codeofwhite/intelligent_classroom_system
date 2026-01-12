import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
from flask import Flask, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# 使用你认为流畅的模型
model = YOLO("back_end/models/yolov8n_openvino_model/", task='detect')

# 全局变量记录学生座位信息
current_students_tracking = {}
video_source = "http://10.231.41.147:8080/video"

def generate_frames():
    camera = cv2.VideoCapture(video_source)
    # 核心优化 1：强制设置低缓冲区，解决“延迟大”的问题
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        # 核心优化 2：连续读取，只处理最后一帧（丢弃过时帧）
        # 如果处理速度跟不上，这能保证你看到的永远是“现在”的画面
        for _ in range(5): 
            camera.grab() 
            
        success, frame = camera.retrieve()
        if not success: break
        
        try:
            # 1. 旋转画面
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # 2. 使用 ByteTrack (比默认追踪器快很多)
            # imgsz=320 进一步提升推理速度
            results = model.track(
                source=frame, 
                imgsz=320, 
                persist=True, 
                classes=[0], 
                tracker="bytetrack.yaml", # 切换到轻量级追踪器
                verbose=False, 
                half=True, 
                conf=0.3
            )
            
            # 3. 手动绘制（比 .plot() 更轻量，且能自定义信息）
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    # 判定座位
                    seat = "A" if (x1 + x2) / 2 < 240 else "B"
                    current_students_tracking[track_id] = {"seat": seat}

                    # 绘图
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id} Seat:{seat}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. 推送
            frame_display = cv2.resize(frame, (480, 640))
            _, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 50])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        except Exception as e:
            print(f"Error: {e}")
            continue

@app.route('/api/stats')
def get_stats():
    return jsonify(current_students_tracking)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 使用 threaded=True 允许并发访问 API 和 视频流
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)