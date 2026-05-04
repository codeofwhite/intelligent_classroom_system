"""
蓝图包 - 按功能场景组织 API
"""
from blueprints.realtime import realtime_bp
from blueprints.model_mgmt import model_bp
from blueprints.video import video_bp
from blueprints.reports import reports_bp
from blueprints.face import face_bp
from blueprints.classes import classes_bp
from blueprints.students import students_bp
from blueprints.parents import parents_bp
from blueprints.ai_bp import ai_bp
from blueprints.chat_bp import chat_bp
from blueprints.schedule import schedule_bp

ALL_BLUEPRINTS = [
    realtime_bp,
    model_bp,
    video_bp,
    reports_bp,
    face_bp,
    classes_bp,
    students_bp,
    parents_bp,
    ai_bp,
    chat_bp,
    schedule_bp,
]