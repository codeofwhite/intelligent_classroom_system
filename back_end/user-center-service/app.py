"""
用户中心服务
主入口文件：注册所有蓝图，启动 Flask 应用

API 按功能场景分为以下蓝图：
  - auth     认证登录（/login）
  - teacher  教师端（/teacher-class, /teacher-students）
  - parent   家长端（/parent-children）
"""
from flask import Flask
from flask_cors import CORS

from blueprints import ALL_BLUEPRINTS


def create_app():
    """应用工厂函数"""
    app = Flask(__name__)
    CORS(app)

    # 注册所有蓝图
    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp)

    @app.route('/')
    def index():
        return "User Center Service Running"

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)