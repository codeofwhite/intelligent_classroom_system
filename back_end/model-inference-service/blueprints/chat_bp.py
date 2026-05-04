"""
聊天 蓝图
- 聊天对话
- 获取会话列表
- 获取会话消息
- 删除会话
"""
from flask import Blueprint, request, jsonify

from shared import get_db_connection
from chat_agent import chat_agent_api, get_session_messages, get_teacher_sessions

chat_bp = Blueprint("chat", __name__)


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