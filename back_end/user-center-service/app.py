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
    role = data.get('role')

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            # 1. 验证用户名、密码和角色是否匹配
            # 生产环境应使用 Werkzeug 密码哈希校验，这里演示用明文
            sql_user = "SELECT * FROM users WHERE username = %s AND password = %s AND role = %s"
            cursor.execute(sql_user, (username, password, role))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({"status": "error", "message": "用户名、密码或身份不匹配"}), 401

            # 2. 查询该用户的关联关系
            # 修改了 SQL 使其能查出关联人的姓名和角色
            sql_rel = """
                SELECT u.username, u.role, r.relation_type 
                FROM user_relations r
                JOIN users u ON r.to_user_id = u.id
                WHERE r.from_user_id = %s
            """
            cursor.execute(sql_rel, (user['id'],))
            relations = cursor.fetchall()
            
            # 为了让前端知道“我是谁的家长”，如果自己是家长，则 relations 里的学生就是自己的孩子
            return jsonify({
                "status": "success",
                "user": {
                    "id": user['id'],
                    "username": user['username'],
                    "role": user['role']
                },
                "relations": relations
            })
    finally:
        conn.close()
           
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)