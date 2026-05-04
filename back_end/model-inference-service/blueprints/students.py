"""
学生端 蓝图
- 学生首页信息
- 学生个人信息
- 学生个人统计数据（今日/本周/本月/学期）
- 学生行为汇总
- 我的报告列表
"""
import json
from datetime import date

import pymysql
from flask import Blueprint, request, jsonify

from shared import minio_client, BUCKET_NAME, get_db_connection
from utils import round0

students_bp = Blueprint("students", __name__)


@students_bp.route("/api/student/home", methods=["GET"])
def student_home():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"student_name": "", "class_name": ""})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT u.name AS student_name, c.class_name
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            JOIN classes c ON s.class_code = c.class_code
            WHERE s.student_code=%s
        """, (student_code,))
        info = cursor.fetchone()
        cursor.close()
        db_tmp.close()
        return jsonify(info)
    except Exception as e:
        return jsonify({"student_name": "", "class_name": ""})


@students_bp.route("/api/student/info", methods=["GET"])
def student_info():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"error": "缺少student_code"}), 400

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT u.name AS student_name, s.class_code, c.class_name
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            JOIN classes c ON s.class_code = c.class_code
            WHERE s.student_code=%s
        """, (student_code,))
        info = cursor.fetchone()
        cursor.close()
        db_tmp.close()
        return jsonify(info)
    except Exception as e:
        return jsonify({
            "student_name": "",
            "class_code": "",
            "class_name": ""
        })


@students_bp.route("/api/student/stats", methods=["GET"])
def student_stats():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"error": "缺少student_code"}), 400

    today = date.today()
    first_day_of_month = today.replace(day=1)

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        # 今日
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS focus,
                   IFNULL(SUM(normal_posture),0) AS lookUp,
                   IFNULL(SUM(looking_down),0) AS disturb
            FROM student_reports
            WHERE student_code=%s AND DATE(lesson_time)=CURDATE()
        """, (student_code,))
        day = cursor.fetchone()

        # 本周
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg
            FROM student_reports
            WHERE student_code=%s AND YEARWEEK(lesson_time)=YEARWEEK(NOW())
        """, (student_code,))
        week = cursor.fetchone()

        # 本月
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg,
                   COUNT(*) AS classCount
            FROM student_reports
            WHERE student_code=%s AND DATE(lesson_time)>=%s
        """, (student_code, first_day_of_month))
        month = cursor.fetchone()

        # 学期平均
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg FROM student_reports
            WHERE student_code=%s
        """, (student_code,))
        semester = cursor.fetchone()

        cursor.close()
        db_tmp.close()

        return jsonify({
            "day": {
                "focus": round0(day['focus']),
                "lookUp": round0(day['lookUp']),
                "disturb": round0(day['disturb'])
            },
            "week": {
                "avg": round0(week['avg']),
                "up": 5,
                "bestDay": "周四"
            },
            "month": {
                "avg": round0(month['avg']),
                "progress": 7,
                "classCount": month['classCount']
            },
            "semester": {
                "avg": round0(semester['avg']),
                "level": "A · 优秀" if round0(semester['avg']) >= 85 else "B · 良好"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@students_bp.route("/api/student/behavior", methods=["GET"])
def get_student_behavior():
    class_code = request.args.get("class_code")
    face_id = request.args.get("face_id")

    if not class_code or not face_id:
        return jsonify({"error": "缺少参数"}), 400

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        cursor.execute("""
            SELECT minio_json_path FROM course_reports
            WHERE class_code=%s ORDER BY created_at DESC
        """, (class_code,))
        reports = cursor.fetchall()
        cursor.close()
        db_tmp.close()

        # 汇总该学生所有行为
        total_behaviors = {}
        for r in reports:
            try:
                data = minio_client.get_object(BUCKET_NAME, r["minio_json_path"])
                stats = json.load(data)
                sb = stats.get("student_behaviors", {})
                if face_id in sb:
                    for b, cnt in sb[face_id].items():
                        total_behaviors[b] = total_behaviors.get(b, 0) + cnt
            except:
                continue

        return jsonify({
            "face_id": face_id,
            "behaviors": total_behaviors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@students_bp.route("/api/student/my-reports", methods=["GET"])
def my_reports():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"list": []})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT * FROM student_reports
            WHERE student_code=%s
            ORDER BY lesson_time DESC
        """, (student_code,))
        lst = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": lst})
    except Exception as e:
        return jsonify({"list": []})