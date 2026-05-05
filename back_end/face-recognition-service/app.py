"""
人脸识别签到服务
主入口文件：注册路由，启动 Flask 应用

模块职责：
  - config.py       配置常量（路径、日志文件）
  - face_db.py      人脸库加载、签到逻辑、日志读取
  - video_stream.py 摄像头视频流 + 实时识别
  - face_engine.py  独立 CLI 工具（脱离 Flask 直接运行签到）
"""
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from face_db import load_face_database, get_sign_logs, load_face_database_from_minio
from video_stream import gen_frames

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "人脸识别签到服务已启动"


@app.route('/video_feed')
def video_feed():
    """实时视频流（MJPEG）"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_sign_log')
def get_sign_log():
    """获取签到日志"""
    logs = get_sign_logs()
    return jsonify(logs=logs)


@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    """从 MinIO 重新加载人脸库（供 model-inference-service 远程调用）"""
    try:
        load_face_database_from_minio()
        return jsonify({"status": "ok", "count": len(__import__('face_db').known_names)})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


if __name__ == "__main__":
    # 优先从 MinIO 加载，失败则从本地加载
    try:
        load_face_database_from_minio()
    except Exception as e:
        print(f"MinIO 加载失败，回退到本地: {e}")
        load_face_database()
    app.run(host="0.0.0.0", port=5003, debug=True)
