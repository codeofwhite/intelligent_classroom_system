"""
班级管理 蓝图
- 获取班级列表
- 获取班级学生列表
- 班级专注度排行
"""
import pymysql
from flask import Blueprint, request, jsonify

from shared import get_db_connection
from utils import round0

classes_bp = Blueprint("classes", __name__)


@classes_bp.route("/api/class/list", methods=["GET"])
def get_class_list():
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT id, class_code, class_name FROM classes")
        class_list = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": class_list})
    except Exception as e:
        return jsonify({"list": []}), 500


@classes_bp.route("/api/class/students", methods=["GET"])
def get_class_students():
    class_code = request.args.get("class_code")
    if not class_code:
        return jsonify({"students": []})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT s.student_code, u.name
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            WHERE s.class_code = %s
        """, (class_code,))
        students = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"students": students})
    except:
        return jsonify({"students": []})


@classes_bp.route("/api/class/rank", methods=["GET"])
def class_rank():
    class_code = request.args.get("class_code")
    if not class_code:
        return jsonify({"rank": []})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT 
                u.name AS name,
                IFNULL(AVG(r.focus_rate), 0) AS score
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            LEFT JOIN student_reports r 
                ON s.student_code = r.student_code 
                AND r.class_code = %s
            WHERE s.class_code = %s
            GROUP BY s.student_code, u.name
            HAVING score > 0
            ORDER BY score DESC
            LIMIT 10
        """, (class_code, class_code))
        ranks = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"rank": ranks})
    except Exception as e:
        print("排行错误：", e)
        return jsonify({"rank": []})