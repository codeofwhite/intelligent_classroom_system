"""
聊天 蓝图
- 聊天对话
- 获取会话列表
- 获取会话消息
- 删除会话
"""
import os
from flask import Blueprint, request, jsonify
import pymysql
from dashscope import Generation

from shared import get_db_connection
from chat_agent import chat_agent_api, get_session_messages, get_teacher_sessions

# 数据库配置（从环境变量读取）
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "user_center_db"),
    "charset": "utf8mb4"
}

chat_bp = Blueprint("chat", __name__)


# ------------------------------
# 教师端 AI 对话
# ------------------------------
@chat_bp.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    question = data.get("question", "")
    teacher_code = data.get("teacher_code", "")
    session_id = data.get("session_id", "")
    result = chat_agent_api(question, teacher_code, session_id)
    return jsonify(result)


@chat_bp.route('/api/chat/sessions', methods=['POST'])
def api_chat_sessions():
    data = request.json
    teacher_code = data.get("teacher_code", "")
    sessions = get_teacher_sessions(teacher_code)
    return jsonify({"sessions": sessions})


@chat_bp.route('/api/chat/messages', methods=['POST'])
def api_chat_messages():
    data = request.json
    teacher_code = data.get("teacher_code", "")
    session_id = data.get("session_id", "")
    messages = get_session_messages(teacher_code, session_id)
    return jsonify({"messages": messages})


@chat_bp.route("/api/chat/delete_session", methods=["POST"])
def delete_session():
    try:
        data = request.get_json()
        teacher_code = data.get("teacher_code")
        session_id = data.get("session_id")

        if not teacher_code or not session_id:
            return jsonify({"code": 400, "msg": "参数缺失"}), 400

        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("""
            DELETE FROM chat_sessions
            WHERE teacher_code=%s AND session_id=%s
        """, (teacher_code, session_id))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"code": 200, "msg": "删除成功"})
    except Exception as e:
        print("删除会话错误：", e)
        return jsonify({"code": 500, "msg": "删除失败"}), 500


# ------------------------------
# ✅ 学生端 AI 学习助手（正确写入蓝图）
# ------------------------------
@chat_bp.route("/api/student/ai", methods=["POST"])
def student_ai():
    data = request.get_json()
    question = data.get("question", "")
    student_code = data.get("student_code")

    if not student_code:
        return jsonify({"answer": "请先登录学生账号"})

    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT lesson_time, focus_rate, normal_posture, raised_hand, looking_down, ai_comment
            FROM student_reports
            WHERE student_code = %s
            ORDER BY lesson_time DESC
        """, (student_code,))
        reports = cursor.fetchall()
        cursor.close()
        db.close()
    except Exception as e:
        print("学生AI查询错误:", e)
        return jsonify({"answer": "获取数据失败"})

    if not reports:
        return jsonify({"answer": "你还没有课堂报告哦～"})

    prompt = f"""
你是学生的AI学习助教，语气亲切、可爱、鼓励。
学生问：{question}

学生课堂数据：
{reports}

请用简短、友好、鼓励的话回答。
"""

    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = resp.output.text.strip()
    return jsonify({"answer": answer})