"""
教师端 蓝图
- 教师班级信息
- 教师学生列表
"""
from flask import Blueprint, request, jsonify

from config import get_db_conn

teacher_bp = Blueprint("teacher", __name__)


@teacher_bp.route('/teacher-class', methods=['POST'])
def teacher_class():
    data = request.json
    user_code = data.get('user_code')

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT t.subject, c.class_name, c.class_code
                FROM teachers t
                JOIN classes c ON t.class_code = c.class_code
                WHERE t.user_code = %s
            """
            cursor.execute(sql, (user_code,))
            teacher = cursor.fetchone()

            class_code = teacher['class_code']
            subject = teacher['subject']
            class_name = teacher['class_name']

            sql = """
                SELECT u.name, s.gender
                FROM students s
                JOIN users u ON s.user_code = u.user_code
                WHERE s.class_code = %s
            """
            cursor.execute(sql, (class_code,))
            students = cursor.fetchall()

            return jsonify({
                "class_name": class_name,
                "class_code": class_code,
                "subject": subject,
                "student_count": len(students),
                "students": students
            })

    finally:
        conn.close()


@teacher_bp.route('/teacher-students', methods=['POST'])
def teacher_students():
    data = request.json
    user_code = data.get('user_code')

    if not user_code:
        return jsonify({"class_name": "", "students": []})

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            # 1. 查老师 + 班级
            sql = """
                SELECT c.class_code, c.class_name
                FROM teachers t
                JOIN classes c ON t.class_code = c.class_code
                WHERE t.user_code = %s
            """
            cursor.execute(sql, (user_code,))
            teacher = cursor.fetchone()

            if not teacher:
                return jsonify({"class_name": "", "students": []})

            class_code = teacher['class_code']
            class_name = teacher['class_name']

            # 2. 查班级学生
            sql = """
                SELECT
                    s.student_code,
                    u.name as student_name,
                    s.gender,
                    up.name as parent_name,
                    up.phone as parent_phone
                FROM students s
                JOIN users u ON s.user_code = u.user_code
                LEFT JOIN parents p ON s.student_code = p.student_code
                LEFT JOIN users up ON p.user_code = up.user_code
                WHERE s.class_code = %s
            """
            cursor.execute(sql, (class_code,))
            students = cursor.fetchall()

            return jsonify({
                "class_name": class_name,
                "class_code": class_code,
                "students": students
            })
    finally:
        conn.close()
