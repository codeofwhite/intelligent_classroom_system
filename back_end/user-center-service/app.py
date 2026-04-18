# user-center-service/app.py
from flask import Flask, request, jsonify
import pymysql
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_db_conn():
    return pymysql.connect(
            host=os.getenv('DB_HOST', 'user-db'),
            user=os.getenv('DB_USER', 'root'),          # 默认用 root
            password=os.getenv('DB_PASSWORD', 'password123'), # 默认密码
            database=os.getenv('DB_NAME', 'user_center_db'),
            port=int(os.getenv('DB_PORT', '3306')),
            cursorclass=pymysql.cursors.DictCursor
    )

# 健康检查
@app.route('/')
def index():
    return "User Center Service Running"

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
            # 1. 校验用户
            sql = """
                SELECT id, username, name, role 
                FROM users 
                WHERE username = %s AND password = %s AND role = %s
            """
            cursor.execute(sql, (username, password, role))
            user = cursor.fetchone()

            if not user:
                return jsonify({"status": "error", "message": "账号或密码错误"}), 401

            user_id = user["id"]
            relations = []

            # =============================================
            # 2. 根据角色查询【关联数据】
            # =============================================
            if role == "student":
                # 学生 → 查班级
                sql = """
                    SELECT c.class_name, c.grade
                    FROM students s
                    JOIN classes c ON s.class_id = c.id
                    WHERE s.user_id = %s
                """
                cursor.execute(sql, (user_id,))
                class_info = cursor.fetchone()
                relations.append({
                    "type": "class",
                    "class_name": class_info["class_name"],
                    "grade": class_info["grade"]
                })

            elif role == "parent":
                # 家长 → 查关联的孩子
                sql = """
                    SELECT u.name as username
                    FROM parents p
                    JOIN students s ON p.student_id = s.id
                    JOIN users u ON s.user_id = u.id
                    WHERE p.user_id = %s
                """
                cursor.execute(sql, (user_id,))
                children = cursor.fetchall()
                for child in children:
                    relations.append({
                        "type": "student",
                        "username": child["username"]
                    })

            elif role == "teacher":
                # 老师 → 查授课班级
                sql = """
                    SELECT c.class_name, c.grade
                    FROM teachers t
                    JOIN classes c ON t.class_id = c.id
                    WHERE t.user_id = %s
                """
                cursor.execute(sql, (user_id,))
                class_info = cursor.fetchone()
                relations.append({
                    "type": "class",
                    "class_name": class_info["class_name"],
                    "grade": class_info["grade"]
                })

            # =============================================
            # 3. 返回给前端：用户信息 + 关联关系
            # =============================================
            return jsonify({
                "status": "success",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "name": user["name"],
                    "role": user["role"]
                },
                "relations": relations  # 前端要用的关联数据
            })

    finally:
        conn.close()
        
@app.route('/teacher-class', methods=['POST'])
def teacher_class():
    data = request.json
    user_id = data.get('user_id')

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            # 1. 获取老师的班级、科目
            sql = """
                SELECT t.subject, c.class_name, c.id as class_id
                FROM teachers t
                JOIN classes c ON t.class_id = c.id
                WHERE t.user_id = %s
            """
            cursor.execute(sql, (user_id,))
            teacher = cursor.fetchone()

            class_id = teacher['class_id']
            subject = teacher['subject']
            class_name = teacher['class_name']

            # 2. 获取本班学生
            sql = """
                SELECT u.name, s.gender
                FROM students s
                JOIN users u ON s.user_id = u.id
                WHERE s.class_id = %s
            """
            cursor.execute(sql, (class_id,))
            students = cursor.fetchall()

            return jsonify({
                "class_name": class_name,
                "subject": subject,
                "student_count": len(students),
                "students": students
            })

    finally:
        conn.close()        

@app.route('/teacher-students', methods=['POST'])
def teacher_students():
    data = request.json
    user_id = data.get('user_id')

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            # 1. 获取老师负责的班级
            sql = """
                SELECT c.class_name, c.id as class_id
                FROM teachers t
                JOIN classes c ON t.class_id = c.id
                WHERE t.user_id = %s
            """
            cursor.execute(sql, (user_id,))
            teacher = cursor.fetchone()
            class_id = teacher['class_id']
            class_name = teacher['class_name']

            # 2. 查询本班学生 + 家长信息（关键关联！）
            sql = """
                SELECT
                    s.id as student_id,
                    u.name as student_name,
                    s.gender,
                    up.name as parent_name,
                    up.phone as parent_phone
                FROM students s
                JOIN users u ON s.user_id = u.id
                LEFT JOIN parents p ON s.id = p.student_id
                LEFT JOIN users up ON p.user_id = up.id
                WHERE s.class_id = %s
            """
            cursor.execute(sql, (class_id,))
            students = cursor.fetchall()

            return jsonify({
                "class_name": class_name,
                "students": students
            })
    finally:
        conn.close()
       
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)