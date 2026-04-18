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

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role') # teacher / student / parent

    if not username or not password or not role:
        return jsonify({"status": "error", "message": "参数不全"}), 400

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT id, username, name, role 
                FROM users 
                WHERE username = %s AND password = %s AND role = %s
            """            
            cursor.execute(sql, (username, password, role))            
            user = cursor.fetchone()
            
            if not user:
                return jsonify({"status": "error", "message": "用户名或密码错误"}), 401
            
            # 返回教师信息（前端需要）
            return jsonify({
                "status": "success",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "name": user["name"],
                    "role": user["role"]
                }
            })
            
    finally:
        conn.close()
   
# 健康检查
@app.route('/')
def index():
    return "User Center Service Running"
           
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)