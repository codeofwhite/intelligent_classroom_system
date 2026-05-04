"""
课程表 蓝图
- 教师课程表查询
"""
import pymysql
from flask import Blueprint, request, jsonify

from shared import get_db_connection

schedule_bp = Blueprint("schedule", __name__)


@schedule_bp.route('/api/teacher/course_schedule', methods=['POST'])
def api_course_schedule():
    data = request.json
    teacher_code = data.get("teacher_code", "")
    try:
        db = get_db_connection()
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT week_day, section, class_name, course_name, classroom
            FROM teacher_course_schedule
            WHERE teacher_code=%s ORDER BY week_day, section
        """, (teacher_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return jsonify({"list": rows})
    except:
        return jsonify({"list": []})