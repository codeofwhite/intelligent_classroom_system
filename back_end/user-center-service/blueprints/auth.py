"""
认证 蓝图
- 用户登录（学生/教师/家长多角色）
"""
from flask import Blueprint, request, jsonify

from config import get_db_conn

auth_bp = Blueprint("auth", __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')

    if not username or not password or not role:
        return jsonify({"status": "error", "message": "参数不全"}), 400

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT id, user_code, username, name, role 
                FROM users 
                WHERE username = %s AND password = %s AND role = %s
            """
            cursor.execute(sql, (username, password, role))
            user = cursor.fetchone()

            if not user:
                return jsonify({"status": "error", "message": "账号或密码错误"}), 401

            user_code = user["user_code"]
            relations = []
            student_code = None
            teacher_code = None
            parent_code = None

            if role == "student":
                sql = """
                    SELECT s.student_code, c.class_name, c.grade
                    FROM students s
                    JOIN classes c ON s.class_code = c.class_code
                    WHERE s.user_code = %s
                """
                cursor.execute(sql, (user_code,))
                row = cursor.fetchone()
                if row:
                    student_code = row["student_code"]
                    relations.append({
                        "type": "class",
                        "class_name": row["class_name"],
                        "grade": row["grade"]
                    })

            elif role == "parent":
                sql = """
                    SELECT parent_code FROM parents WHERE user_code = %s
                """
                cursor.execute(sql, (user_code,))
                row = cursor.fetchone()
                if row:
                    parent_code = row["parent_code"]

            elif role == "teacher":
                sql = """
                    SELECT t.teacher_code, c.class_name, c.grade
                    FROM teachers t
                    JOIN classes c ON t.class_code = c.class_code
                    WHERE t.user_code = %s
                """
                cursor.execute(sql, (user_code,))
                row = cursor.fetchone()
                if row:
                    teacher_code = row["teacher_code"]
                    relations.append({
                        "type": "class",
                        "class_name": row["class_name"],
                        "grade": row["grade"]
                    })

            return jsonify({
                "status": "success",
                "user": {
                    "id": user["id"],
                    "user_code": user["user_code"],
                    "username": user["username"],
                    "name": user["name"],
                    "role": user["role"],
                    "student_code": student_code,
                    "parent_code": parent_code,
                    "teacher_code": teacher_code
                },
                "relations": relations
            })

    finally:
        conn.close()