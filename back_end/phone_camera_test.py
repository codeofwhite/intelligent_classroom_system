import os
import cv2
import uuid
import tempfile
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from minio import Minio
from datetime import datetime

app = Flask(__name__)
CORS(app) # 解决跨域问题

MODELS_DIR = "models"
# 获取目录下所有以 _openvino_model 结尾的文件夹
available_models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
current_model_name = available_models[0] if available_models else None
model = YOLO(os.path.join(MODELS_DIR, current_model_name), task='detect') if current_model_name else None

# --- 1. 配置 MinIO 连接 ---
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)
BUCKET_NAME = "video-bucket"

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载导出的 OpenVINO 模型
model = YOLO("models/best_last_openvino_model/", task='detect')

video_source = "http://192.168.1.196:8080/video" 

def generate_frames():
    camera = cv2.VideoCapture(video_source)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # 1. 旋转画面
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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
        frame_display = cv2.resize(annotated_frame, (640, 480))
        
        # 4. 编码与推送
        ret, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/get_models', methods=['GET'])
def get_models():
    # 重新扫描文件夹，确保新增模型能看到
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return jsonify({"models": models, "current": current_model_name})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global model, current_model_name
    target_model = request.json.get('model_name')
    
    if target_model not in os.listdir(MODELS_DIR):
        return jsonify({"status": "error", "msg": "模型文件不存在"}), 404
    
    try:
        # 重新加载模型
        new_path = os.path.join(MODELS_DIR, target_model)
        model = YOLO(new_path, task='detect')
        current_model_name = target_model
        print(f"Successfully switched to model: {current_model_name}")
        return jsonify({"status": "success", "current": current_model_name})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 上传视频接口，前端通过 POST 请求上传视频文件
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    file = request.files['video']
    ext = ".mp4" # 强制输出为 mp4 方便网页播放
    unique_filename = f"{uuid.uuid4()}{ext}"
    
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, "input_" + unique_filename)
            output_path = os.path.join(tmpdirname, "output_" + unique_filename)
            file.save(input_path)

            # 1. 使用 OpenCV 读取视频获取属性
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            
            # 2. 定义写入器 (使用 mp4v 或 avc1 编码)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 3. 逐帧推理并写入 (参考你给的代码逻辑)
            # 使用 stream=True 节省内存
            results = model.predict(source=input_path, stream=True, conf=0.25, verbose=False)
            
            for result in results:
                # 获取渲染了检测框的帧
                annotated_frame = result.plot()
                out.write(annotated_frame)
            
            # 记得一定要释放资源，否则文件会被占用无法上传
            cap.release()
            out.release()

            # 4. 检查输出文件是否存在且有大小
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("视频处理失败，生成的视频文件为空")

            # 在接口内生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:6]
            # 最终存储在 MinIO 的名字：20260410_0055_classroom_a1b2.mp4
            object_name = f"processed/{timestamp}_classroom_{unique_id}.mp4"
            minio_client.fput_object(BUCKET_NAME, object_name, output_path)
            
            # 6. 生成签名链接返回给前端
            download_url = minio_client.get_presigned_url("GET", BUCKET_NAME, object_name)

            return jsonify({
                "status": "success", 
                "video_url": download_url,
                "msg": "处理完成"
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/list_videos', methods=['GET'])
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
        print(f"List error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)