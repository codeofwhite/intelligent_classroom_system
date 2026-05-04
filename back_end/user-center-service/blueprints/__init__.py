"""
蓝图包 - 按功能场景组织 API
"""
from blueprints.auth import auth_bp
from blueprints.teacher import teacher_bp
from blueprints.parent import parent_bp

ALL_BLUEPRINTS = [
    auth_bp,
    teacher_bp,
    parent_bp,
]