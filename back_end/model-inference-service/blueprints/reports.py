"""
课堂报告 蓝图（教师端）
- 教师获取报告列表
- 获取单节课详细分析
- 生成并保存AI分析
- 获取报告中的face_ids
- 获取报告中的学生列表
- 绑定track_id与学生
- 保存/查询/删除学生个人报告
"""
import io
import json
from datetime import datetime

from flask import Blueprint, request, jsonify

from shared import minio_client, BUCKET_NAME, get_db_connection

reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/api/teacher/reports", methods=["GET"])
def teacher_reports():
    teacher_code = request.args.get("teacher_code")
    if not teacher_code:
        return jsonify([])

    db_temp = get_db_connection()
    cursor = db_temp.cursor()
    cursor.execute("""
        SELECT cr.*, c.class_name
        FROM course_reports cr
        JOIN classes c ON cr.class_code = c.class_code
        WHERE cr.teacher_code = %s
        ORDER BY cr.created_at DESC
    """, (teacher_code,))

    # 获取列名，手动转为字典列表
    columns = [desc[0] for desc in cursor.description]
    data = [dict(zip(columns, row)) for row in cursor.fetchall()]

    cursor.close()
    db_temp.close()

    response = jsonify(data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@reports_bp.route("/api/report/detail", methods=["GET"])
def report_detail():
    try:
        report_id = request.args.get("id")

        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("""
            SELECT cr.*, c.class_name
            FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code
            WHERE cr.id=%s
        """, (report_id,))

        columns = [desc[0] for desc in cursor.description]
        report_row = cursor.fetchone()
        report = dict(zip(columns, report_row)) if report_row else {}
        cursor.close()
        db_tmp.close()

        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.loads(data.data)

        ai_text = ""
        try:
            ai_path = report['minio_json_path'].replace("stats.json", "ai_report.md")
            ai_obj = minio_client.get_object(BUCKET_NAME, ai_path)
            ai_text = ai_obj.read().decode("utf-8")
        except:
            ai_text = ""

        return jsonify({
            "report": report,
            "statistics": stats,
            "ai_analysis": ai_text
        })

    except Exception as e:
        print("DETAIL ERROR:", e)
        return jsonify({
            "report": {},
            "statistics": {},
            "ai_analysis": ""
        }), 500


@reports_bp.route("/api/generate_and_save_ai", methods=["POST"])
def generate_and_save_ai():
    try:
        report_id = request.json.get("id")

        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("""
            SELECT cr.*, c.class_name, cr.teacher_code FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code WHERE cr.id=%s
        """, (report_id,))

        columns = [desc[0] for desc in cursor.description]
        report_row = cursor.fetchone()
        report = dict(zip(columns, report_row)) if report_row else None
        cursor.close()
        db_tmp.close()

        if not report:
            return jsonify({"error": "报告不存在"}), 404

        # 读行为统计
        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.load(data)

        # 关键帧
        key_frame_path = report.get("minio_keyframe_path", "")

        # 生成 AI
        from ai_agent import analyze_class_report
        ai_text = analyze_class_report(
            behavior_data=stats["behavior_counts"],
            class_info={
                "class_name": report["class_name"],
                "lesson_section": report["lesson_section"]
            },
            teacher_code=report["teacher_code"],
            course_name=report.get("course_name", "课堂行为分析"),
            frame_path=key_frame_path
        )

        # 保存
        ai_path = report['minio_json_path'].replace("stats.json", "ai_report.md")
        minio_client.put_object(
            BUCKET_NAME,
            ai_path,
            io.BytesIO(ai_text.encode("utf-8")),
            length=len(ai_text.encode("utf-8")),
            content_type="text/markdown"
        )

        return jsonify({"ai_analysis": ai_text})

    except Exception as e:
        print("AI SAVE ERROR:", e)
        return jsonify({"error": str(e)}), 500


@reports_bp.route("/api/report/face_ids", methods=["GET"])
def get_report_face_ids():
    report_id = request.args.get("id")
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("SELECT minio_json_path FROM course_reports WHERE id=%s", (report_id,))

        columns = [desc[0] for desc in cursor.description]
        report_row = cursor.fetchone()
        report = dict(zip(columns, report_row)) if report_row else {}
        cursor.close()
        db_tmp.close()

        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.loads(data.data)
        face_ids = stats.get("face_ids", [])
        return jsonify({"face_ids": face_ids})
    except:
        return jsonify({"face_ids": []})


@reports_bp.route("/api/report/students", methods=["GET"])
def get_report_students():
    report_id = request.args.get("id")
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("SELECT minio_json_path FROM course_reports WHERE id=%s", (report_id,))

        columns = [desc[0] for desc in cursor.description]
        report_row = cursor.fetchone()
        report = dict(zip(columns, report_row)) if report_row else {}
        cursor.close()
        db_tmp.close()

        data = minio_client.get_object(BUCKET_NAME, report['minio_json_path'])
        stats = json.loads(data.data)
        student_ids = list(stats.get("student_behaviors", {}).keys())
        return jsonify({"student_ids": student_ids})
    except:
        return jsonify({"student_ids": []})


@reports_bp.route("/api/report/bind_student", methods=["POST"])
def bind_student():
    report_id = request.json.get("report_id")
    track_id = request.json.get("track_id")
    student_name = request.json.get("student_name")

    db_tmp = get_db_connection()
    cursor = db_tmp.cursor()
    cursor.execute(
        "INSERT INTO report_student_mapping (report_id, track_id, student_name) VALUES (%s,%s,%s)",
        (report_id, track_id, student_name)
    )
    db_tmp.commit()
    cursor.close()
    db_tmp.close()
    return jsonify({"status": "ok"})


@reports_bp.route("/api/report/delete", methods=["POST"])
def delete_report():
    try:
        report_id = request.json.get("report_id")
        if not report_id:
            return jsonify({"error": "缺少 report_id"}), 400

        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("SELECT * FROM course_reports WHERE id=%s", (report_id,))

        columns = [desc[0] for desc in cursor.description]
        report_row = cursor.fetchone()
        report = dict(zip(columns, report_row)) if report_row else None

        if not report:
            cursor.close()
            db_tmp.close()
            return jsonify({"error": "报告不存在"}), 404

        # 删除 MinIO 文件
        try:
            minio_client.remove_object(BUCKET_NAME, report["minio_video_path"])
            minio_client.remove_object(BUCKET_NAME, report["minio_json_path"])
            if report.get("minio_keyframe_path"):
                minio_client.remove_object(BUCKET_NAME, report["minio_keyframe_path"])
        except:
            pass

        # 删除数据库记录
        cursor.execute("DELETE FROM course_reports WHERE id=%s", (report_id,))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()

        return jsonify({"msg": "删除成功"})

    except Exception as e:
        print("删除报告错误：", e)
        return jsonify({"error": str(e)}), 500


@reports_bp.route("/api/report/save", methods=["POST"])
def save_report():
    d = request.json
    db_tmp = get_db_connection()
    cursor = db_tmp.cursor()
    cursor.execute("""
        INSERT INTO student_reports
        (student_code, class_code, lesson_time, normal_posture, raised_hand, looking_down, focus_rate, ai_comment, teacher_score, teacher_comment)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        d["student_code"],
        d["class_code"], d["lesson_time"],
        d["normal_posture"], d["raised_hand"], d["looking_down"], d["focus_rate"],
        d["ai_comment"], d["teacher_score"], d["teacher_comment"]
    ))
    db_tmp.commit()
    cursor.close()
    db_tmp.close()
    return jsonify({"msg": "保存成功"})


@reports_bp.route("/api/report/history", methods=["GET"])
def report_history():
    student_code = request.args.get("student_code")
    class_code = request.args.get("class_code")
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("""
            SELECT * FROM student_reports
            WHERE student_code=%s AND class_code=%s
            ORDER BY lesson_time DESC
        """, (student_code, class_code))

        columns = [desc[0] for desc in cursor.description]
        lst = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        db_tmp.close()
        return jsonify({"list": lst})
    except Exception as e:
        return jsonify({"list": []})


@reports_bp.route("/api/report/list", methods=["GET"])
def report_list():
    student_code = request.args.get("student_code")
    class_code = request.args.get("class_code")
    db_tmp = get_db_connection()
    cursor = db_tmp.cursor()
    cursor.execute("""
        SELECT * FROM student_reports
        WHERE student_code=%s AND class_code=%s
        ORDER BY lesson_time DESC
    """, (student_code, class_code))

    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    db_tmp.close()
    return jsonify({"list": rows})