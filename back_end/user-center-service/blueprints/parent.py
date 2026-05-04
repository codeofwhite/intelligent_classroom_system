"""
家长端 蓝图
- 家长获取绑定的孩子信息
"""
from flask import Blueprint, request, jsonify

from config import get_db_conn

parent_bp = Blueprint("parent", __name__)


@parent_bp.route('/parent-children', methods=['POST'])
def parent_children():
    data = request.json
    user_code = data.get('user_code')

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT s.student_code, u.name AS student_name
                FROM parents p
                JOIN students s ON p.student_code = s.student_code
                JOIN users u ON s.user_code = u.user_code
                WHERE p.user_code = %s
            """
            cursor.execute(sql, (user_code,))
            children = cursor.fetchall()

            return jsonify({
                "children": children
            })
    finally:
        conn.close()