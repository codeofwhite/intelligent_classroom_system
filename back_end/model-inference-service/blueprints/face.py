"""
人脸管理 蓝图
- 根据student_code查询face_id
- 绑定/解除 face_id 与学生
- 获取班级人脸映射
- 获取人脸列表
- 批量导入绑定
- 人脸图片上传/删除/列表（MinIO统一存储）
- 缓存刷新通知
"""
import os
import io
import uuid
import pymysql
import requests
from flask import Blueprint, request, jsonify

from config import DB_CONFIG
from shared import get_db_connection, minio_client

face_bp = Blueprint("face", __name__)

FACE_BUCKET = "face-images"

# ========================
# 人脸图片管理（MinIO）
# ========================

@face_bp.route("/api/face/upload_image", methods=["POST"])
def upload_face_image():
    """上传人脸照片到 MinIO，同时写入 face_images 表"""
    if 'photo' not in request.files:
        return jsonify({"code": 400, "msg": "请选择照片"}), 400

    student_code = request.form.get("student_code", "").strip()
    class_code = request.form.get("class_code", "").strip() or "1"  # 默认 class_code=1
    if not student_code:
        return jsonify({"code": 400, "msg": "缺少 student_code"}), 400

    file = request.files['photo']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        return jsonify({"code": 400, "msg": "仅支持 jpg/png 格式"}), 400

    filename = f"{uuid.uuid4().hex[:8]}{ext}"
    minio_path = f"{class_code}/{student_code}/{filename}"
    file_data = file.read()

    try:
        minio_client.put_object(
            FACE_BUCKET, minio_path,
            io.BytesIO(file_data),
            length=len(file_data),
            content_type=f"image/{ext.replace('.', '')}"
        )

        db_tmp = get_db_connection()
        cursor = db_tmp.cursor()
        cursor.execute(
            "INSERT INTO face_images (student_code, class_code, minio_path) VALUES (%s, %s, %s)",
            (student_code, class_code, minio_path)
        )
        db_tmp.commit()
        img_id = cursor.lastrowid
        cursor.close()
        db_tmp.close()

        # 尝试同步到 face_lib（model-inference-service 本地）
        _sync_to_face_lib(student_code, file_data, filename)

        return jsonify({"code": 200, "msg": "上传成功", "id": img_id, "minio_path": minio_path})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@face_bp.route("/api/face/images", methods=["GET"])
def get_face_images():
    """获取某个学生的所有人脸照片"""
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"images": []})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute(
            "SELECT id, student_code, class_code, minio_path, created_at FROM face_images WHERE student_code=%s ORDER BY created_at DESC",
            (student_code,)
        )
        rows = cursor.fetchall()
        cursor.close()
        db_tmp.close()

        # 为每张图片生成预签名URL
        for r in rows:
            try:
                r["url"] = minio_client.presigned_get_object(FACE_BUCKET, r["minio_path"])
            except:
                r["url"] = ""

        return jsonify({"images": rows})
    except Exception as e:
        return jsonify({"images": [], "error": str(e)})


@face_bp.route("/api/face/delete_image", methods=["POST"])
def delete_face_image():
    """删除一张人脸照片"""
    img_id = request.json.get("id")
    if not img_id:
        return jsonify({"code": 400, "msg": "缺少 id"}), 400

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT minio_path FROM face_images WHERE id=%s", (img_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            db_tmp.close()
            return jsonify({"code": 404, "msg": "图片不存在"}), 404

        # 删除 MinIO 文件
        try:
            minio_client.remove_object(FACE_BUCKET, row["minio_path"])
        except:
            pass

        # 删除 DB 记录
        cursor.execute("DELETE FROM face_images WHERE id=%s", (img_id,))
        db_tmp.commit()
        cursor.close()
        db_tmp.close()

        return jsonify({"code": 200, "msg": "删除成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@face_bp.route("/api/face/refresh_cache", methods=["POST"])
def refresh_face_cache():
    """通知两个服务刷新人脸缓存"""
    results = {}

    # 通知 face-recognition-service
    try:
        r = requests.post("http://localhost:5003/reload_faces", timeout=5)
        results["face_service"] = r.json()
    except Exception as e:
        results["face_service"] = {"error": str(e)}

    # 通知 model-inference-service（自己）
    try:
        _sync_all_from_minio()
        results["model_service"] = {"status": "ok"}
    except Exception as e:
        results["model_service"] = {"error": str(e)}

    return jsonify({"code": 200, "results": results})


def _sync_to_face_lib(student_code, file_data, filename):
    """将照片同步到本地 face_lib 目录供 pose 模型使用"""
    face_lib_dir = os.path.join("face_lib", student_code)
    os.makedirs(face_lib_dir, exist_ok=True)
    local_path = os.path.join(face_lib_dir, filename)
    with open(local_path, "wb") as f:
        f.write(file_data)


def _sync_all_from_minio():
    """从 MinIO 下载所有照片到本地 face_lib 和 faces 目录"""
    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT student_code, class_code, minio_path FROM face_images")
        rows = cursor.fetchall()
        cursor.close()
        db_tmp.close()

        for r in rows:
            try:
                resp = minio_client.get_object(FACE_BUCKET, r["minio_path"])
                data = resp.read()

                # 同步到 face_lib（pose 模型用）
                lib_dir = os.path.join("face_lib", r["student_code"])
                os.makedirs(lib_dir, exist_ok=True)
                fname = os.path.basename(r["minio_path"])
                with open(os.path.join(lib_dir, fname), "wb") as f:
                    f.write(data)
            except:
                continue
    except Exception as e:
        print(f"同步人脸缓存失败: {e}")


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