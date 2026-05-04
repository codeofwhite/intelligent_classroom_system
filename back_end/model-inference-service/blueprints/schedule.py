"""
课程表 蓝图
- 教师课程表查询
- 课程表关联课堂报告
"""
import pymysql
from flask import Blueprint, request, jsonify

from shared import get_db_connection

schedule_bp = Blueprint("schedule", __name__)


@schedule_bp.route('/api/teacher/course_schedule', methods=['POST'])
def api_course_schedule():
    """获取课程表（含 class_code），用于前端展示"""
    data = request.json
    teacher_code = data.get("teacher_code", "")
    try:
        db = get_db_connection()
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT tcs.week_day, tcs.section, tcs.class_name,
                   tcs.course_name, tcs.classroom, c.class_code
            FROM teacher_course_schedule tcs
            LEFT JOIN classes c ON tcs.class_name = c.class_name
            WHERE tcs.teacher_code=%s
            ORDER BY tcs.week_day, tcs.section
        """, (teacher_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return jsonify({"list": rows})
    except:
        return jsonify({"list": []})


@schedule_bp.route('/api/teacher/schedule_with_reports', methods=['POST'])
def schedule_with_reports():
    """
    返回课程表 + 每个课程关联的课堂报告列表
    前端可直接知道哪个课程有报告、有几份
    """
    data = request.json
    teacher_code = data.get("teacher_code", "")
    if not teacher_code:
        return jsonify({"schedule": [], "reports": []})

    try:
        db = get_db_connection()
        cursor = db.cursor(pymysql.cursors.DictCursor)

        # 1. 获取课程表（含 class_code）
        cursor.execute("""
            SELECT tcs.week_day, tcs.section, tcs.class_name,
                   tcs.course_name, tcs.classroom, c.class_code
            FROM teacher_course_schedule tcs
            LEFT JOIN classes c ON tcs.class_name = c.class_name
            WHERE tcs.teacher_code=%s
            ORDER BY tcs.week_day, tcs.section
        """, (teacher_code,))
        schedule = cursor.fetchall()

        # 2. 获取该教师所有课堂报告
        cursor.execute("""
            SELECT cr.id, cr.report_code, cr.class_code, cr.lesson_section,
                   cr.created_at, cr.minio_keyframe_path, c.class_name
            FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code
            WHERE cr.teacher_code = %s
            ORDER BY cr.created_at DESC
        """, (teacher_code,))
        reports = cursor.fetchall()

        cursor.close()
        db.close()

        return jsonify({
            "schedule": schedule,
            "reports": reports
        })
    except Exception as e:
        print("课表报告查询失败:", e)
        return jsonify({"schedule": [], "reports": []})
