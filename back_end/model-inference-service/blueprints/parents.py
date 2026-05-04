"""
家长端 蓝图
- 家长首页数据（孩子信息 + 概况）
"""
import pymysql
from flask import Blueprint, request, jsonify

from shared import get_db_connection
from utils import round0

parents_bp = Blueprint("parents", __name__)


@parents_bp.route("/api/parent/home", methods=["GET"])
def parent_home():
    user_code = request.args.get("user_code")
    if not user_code:
        return jsonify({})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        # 1. 获取家长绑定的孩子
        cursor.execute("""
            SELECT p.student_code
            FROM parents p
            WHERE p.user_code=%s
        """, (user_code,))
        parent = cursor.fetchone()
        if not parent:
            cursor.close()
            db_tmp.close()
            return jsonify({})

        student_code = parent["student_code"]

        # 2. 获取孩子信息
        cursor.execute("""
            SELECT u.name AS student_name, c.class_name, s.class_code
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            JOIN classes c ON s.class_code = c.class_code
            WHERE s.student_code=%s
        """, (student_code,))
        student = cursor.fetchone()

        # 3. 今日专注度
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate), 0) AS today_focus
            FROM student_reports
            WHERE student_code=%s AND DATE(lesson_time)=CURDATE()
        """, (student_code,))
        today = cursor.fetchone()

        # 4. 总报告数量
        cursor.execute("""
            SELECT COUNT(*) AS total FROM student_reports
            WHERE student_code=%s
        """, (student_code,))
        total = cursor.fetchone()

        cursor.close()
        db_tmp.close()

        return jsonify({
            "student_name": student["student_name"],
            "class_name": student["class_name"],
            "today_focus": round(float(today["today_focus"])),
            "total_reports": total["total"],
            "student_code": student_code
        })

    except Exception as e:
        print("家长首页错误：", e)
        return jsonify({})