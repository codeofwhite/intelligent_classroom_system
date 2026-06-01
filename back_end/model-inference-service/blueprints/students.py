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

import shared
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

        # 本周每日平均（用于计算最佳日和较上周变化）
        cursor.execute("""
            SELECT DAYOFWEEK(lesson_time) AS dow, IFNULL(AVG(focus_rate),0) AS avg
            FROM student_reports
            WHERE student_code=%s AND YEARWEEK(lesson_time)=YEARWEEK(NOW())
            GROUP BY DAYOFWEEK(lesson_time)
            ORDER BY avg DESC
        """, (student_code,))
        week_days = cursor.fetchall()
        best_day_map = {1:'周日',2:'周一',3:'周二',4:'周三',5:'周四',6:'周五',7:'周六'}
        best_day = best_day_map.get(week_days[0]['dow'], '--') if week_days else '--'

        # 上周平均
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg
            FROM student_reports
            WHERE student_code=%s AND YEARWEEK(lesson_time)=YEARWEEK(NOW())-1
        """, (student_code,))
        last_week = cursor.fetchone()
        week_change = round0(week['avg']) - round0(last_week['avg']) if last_week else 0

        # 上月平均（用于计算进步幅度）
        cursor.execute("""
            SELECT IFNULL(AVG(focus_rate),0) AS avg
            FROM student_reports
            WHERE student_code=%s AND MONTH(lesson_time)=MONTH(DATE_SUB(NOW(), INTERVAL 1 MONTH))
              AND YEAR(lesson_time)=YEAR(DATE_SUB(NOW(), INTERVAL 1 MONTH))
        """, (student_code,))
        last_month = cursor.fetchone()
        month_progress = round0(month['avg']) - round0(last_month['avg']) if last_month else 0

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
                "up": max(0, week_change),
                "bestDay": best_day
            },
            "month": {
                "avg": round0(month['avg']),
                "progress": max(0, month_progress),
                "classCount": month['classCount']
            },
            "semester": {
                "avg": round0(semester['avg']),
                "level": "A · 优秀" if round0(semester['avg']) >= 85 else ("B · 良好" if round0(semester['avg']) >= 70 else "C · 一般")
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

        # 根据当前模型配置计算专注度
        focus_rate = 100
        config = shared.current_model_config
        if config and total_behaviors:
            labels_en = config["labels_en"]
            labels_cn = config["labels_cn"]
            focus_en = set(config["focus"])
            focus_count = 0
            distract_count = 0
            for beh, cnt in total_behaviors.items():
                if beh in labels_cn:
                    idx = labels_cn.index(beh)
                    en_label = labels_en[idx]
                    if en_label in focus_en:
                        focus_count += cnt
                    else:
                        distract_count += cnt
                else:
                    distract_count += cnt
            total = focus_count + distract_count
            if total > 0:
                focus_rate = round(100 * focus_count / total)

        return jsonify({
            "face_id": face_id,
            "behaviors": total_behaviors,
            "focus_rate": focus_rate
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