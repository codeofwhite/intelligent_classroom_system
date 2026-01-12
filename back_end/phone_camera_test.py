import cv2
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

# 加载导出的 OpenVINO 模型
model = YOLO("back_end/models/best_last_openvino_model/", task='detect')

video_source = "http://10.231.41.147:8080/video" 

def generate_frames():
    camera = cv2.VideoCapture(video_source)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # 1. 旋转画面 (变成竖向)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 不再使用手动 resize(640, 640)，而是让 YOLO 内部处理等比例缩放
        # imgsz=640 告诉 YOLO 以 640 为长边缩放，不足的部分会自动补黑边
        results = model.predict(
            source=frame, 
            imgsz=640,      # 保持 640 适配模型
            verbose=False, 
            half=True,      # Intel GPU 加速必备
            conf=0.25       # 置信度阈值，如果漏检可以调低到 0.2
        )
        
        # 2. 获取渲染后的画面
        annotated_frame = results[0].plot()

        # 3. 图像后处理 (网页显示尺寸)
        # 保持旋转后的比例显示：480宽，640高
        frame_display = cv2.resize(annotated_frame, (480, 640))
        
        # 4. 编码与推送
        ret, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)