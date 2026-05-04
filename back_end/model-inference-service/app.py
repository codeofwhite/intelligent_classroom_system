"""
智能课堂系统 - 模型推理服务
主入口文件：注册所有蓝图，启动 Flask 应用

API 按功能场景分为以下蓝图：
  - realtime     实时监测与录制（/start_record, /stop_record, /get_realtime_stats, /get_record_status, /video_feed）
  - model_mgmt   模型管理     （/get_models, /switch_model）
  - video        视频上传分析  （/upload_video, /list_videos, /get_history_stat, /get_video_url）
  - reports      课堂报告     （/api/teacher/reports, /api/report/*, /api/generate_and_save_ai）
  - face         人脸管理     （/api/face/*）
  - classes      班级管理     （/api/class/*）
  - students     学生端       （/api/student/*）
  - parents      家长端       （/api/parent/*）
  - ai           AI 分析      （/api/ai/*）
  - chat         聊天助手     （/api/chat/*）
  - schedule     课程表       （/api/teacher/course_schedule）
"""
from flask import Flask
from flask_cors import CORS

from blueprints import ALL_BLUEPRINTS


def create_app():
    """应用工厂函数"""
    app = Flask(__name__)
    app.config['DEBUG'] = True
    CORS(app)

    # 注册所有蓝图
    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)