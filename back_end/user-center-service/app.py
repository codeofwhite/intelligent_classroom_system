from flask import Flask, request, jsonify
import pymysql
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_db_conn():
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'user-db'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', 'password123'),
        database=os.getenv('DB_NAME', 'user_center_db'),
        port=int(os.getenv('DB_PORT', 3306)),
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/')
def index():
    return "User Center Service Running"

# ==============================
# 登录
# ==============================
@app.route('/login', methods=['POST'])
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

# ==============================
# 家长获取孩子
# ==============================
@app.route('/parent-children', methods=['POST'])
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

# ==============================
# 老师班级
# ==============================
@app.route('/teacher-class', methods=['POST'])
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
                "subject": subject,
                "student_count": len(students),
                "students": students
            })

    finally:
        conn.close()

# ==============================
# 老师学生列表
# ==============================
@app.route('/teacher-students', methods=['POST'])
def teacher_students():
    data = request.json
    user_code = data.get('user_code')

    # 🔥 加判空，防止崩溃
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

            # 🔥 查不到就直接返回空
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
                "students": students
            })
    finally:
        conn.close()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)