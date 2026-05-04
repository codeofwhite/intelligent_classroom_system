"""
人脸管理 蓝图
- 根据student_code查询face_id
- 绑定/解除 face_id 与学生
- 获取班级人脸映射
- 获取人脸列表
- 批量导入绑定
"""
import pymysql
from flask import Blueprint, request, jsonify

from config import DB_CONFIG
from shared import get_db_connection

face_bp = Blueprint("face", __name__)


@face_bp.route("/api/face/by_student", methods=["GET"])
def get_face_by_student():
    student_code = request.args.get("student_code")
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT face_id FROM face_student_mapping WHERE student_code=%s", (student_code,))
        row = cursor.fetchone()
        cursor.close()
        db_tmp.close()
        return jsonify({"face_id": row["face_id"] if row else None})
    except:
        return jsonify({"face_id": None})


@face_bp.route("/api/face/mapping", methods=["GET"])
def get_face_mapping():
    class_code = request.args.get("class_code")
    if not class_code:
        return jsonify({"map": {}})
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT face_id, s.student_code, fsm.student_name
            FROM face_student_mapping fsm
            JOIN students s ON fsm.student_code = s.student_code
            WHERE fsm.class_code=%s
        """, (class_code,))
        rows = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        mapping = {r["face_id"]: {"id": r["student_code"], "name": r["student_name"]} for r in rows}
        return jsonify({"map": mapping})
    except:
        return jsonify({"map": {}})


@face_bp.route("/api/face/bind", methods=["POST"])
def api_face_bind():
    data = request.json
    face_id = data.get("face_id")
    student_code = data.get("student_code")
    student_name = data.get("student_name")
    class_code = data.get("class_code", None)

    if not face_id or not student_name:
        return jsonify({"code": 400, "msg": "参数错误"}), 400

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("""
            INSERT INTO face_student_mapping
            (face_id, student_code, student_name, class_code)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                student_code = %s,
                student_name = %s,
                class_code = %s
        """, (
            face_id, student_code, student_name, class_code,
            student_code, student_name, class_code
        ))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()
        return jsonify({"code": 200, "msg": "绑定成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@face_bp.route("/api/face/list", methods=["GET"])
def api_face_list():
    class_code = request.args.get("class_code", None)
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        sql = "SELECT face_id, student_name, class_code FROM face_student_mapping"
        if class_code:
            sql += " WHERE class_code=%s"
            cursor.execute(sql, (class_code,))
        else:
            cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        db_tmp.close()
        return jsonify({"list": rows})
    except:
        return jsonify({"list": []})


@face_bp.route("/api/face/batch_import", methods=["POST"])
def api_face_batch_import():
    if 'file' not in request.files:
        return jsonify({"code": 400, "msg": "请上传文件"}), 400

    file = request.files['file']
    class_code = request.form.get("class_code", None)
    try:
        import csv
        reader = csv.DictReader(file.read().decode("utf-8").splitlines())
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        for row in reader:
            face_id = row.get("face_id")
            student_name = row.get("student_name")
            if face_id and student_name:
                cursor.execute("""
                    INSERT INTO face_student_mapping (face_id, student_name, class_code)
                    VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE student_name=%s
                """, (face_id, student_name, class_code, student_name))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()
        return jsonify({"code": 200, "msg": "批量导入成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@face_bp.route("/api/face/unbind", methods=["POST"])
def api_face_unbind():
    face_id = request.json.get("face_id")
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute("DELETE FROM face_student_mapping WHERE face_id=%s", (face_id,))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()
        return jsonify({"code": 200, "msg": "解除成功"})
    except:
        return jsonify({"code": 500, "msg": "失败"}), 500