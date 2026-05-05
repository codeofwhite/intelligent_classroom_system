"""
迁移脚本：将本地 face_lib/ 目录下的人脸照片上传到 MinIO face-images bucket
同时写入 face_images 表

用法：python migrate_faces_to_minio.py
"""
import os
import io
import pymysql
from minio import Minio

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password123",
    "database": "user_center_db",
    "charset": "utf8mb4"
}

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS = "admin"
MINIO_SECRET = "password123"
FACE_BUCKET = "face-images"

FACE_LIB_DIR = "face_lib"  # model-inference-service 下的 face_lib 目录


def migrate():
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=False
    )

    db = pymysql.connect(**DB_CONFIG)
    cursor = db.cursor()

    total = 0
    skipped = 0
    uploaded = 0

    if not os.path.exists(FACE_LIB_DIR):
        print(f"❌ {FACE_LIB_DIR} 目录不存在")
        return

    for student_code in os.listdir(FACE_LIB_DIR):
        stu_dir = os.path.join(FACE_LIB_DIR, student_code)
        if not os.path.isdir(stu_dir):
            continue

        # 默认 class_code=1（你可以根据实际情况修改）
        class_code = "1"

        for fname in os.listdir(stu_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total += 1
            local_path = os.path.join(stu_dir, fname)
            minio_path = f"{class_code}/{student_code}/{fname}"

            # 检查是否已存在
            cursor.execute(
                "SELECT id FROM face_images WHERE minio_path=%s", (minio_path,)
            )
            if cursor.fetchone():
                skipped += 1
                continue

            # 上传到 MinIO
            try:
                with open(local_path, "rb") as f:
                    data = f.read()
                minio_client.put_object(
                    FACE_BUCKET, minio_path,
                    io.BytesIO(data),
                    length=len(data),
                    content_type="image/jpeg"
                )
            except Exception as e:
                print(f"  ❌ 上传失败 {local_path}: {e}")
                continue

            # 写入 DB
            try:
                cursor.execute(
                    "INSERT INTO face_images (student_code, class_code, minio_path) VALUES (%s, %s, %s)",
                    (student_code, class_code, minio_path)
                )
                uploaded += 1
            except Exception as e:
                print(f"  ❌ DB写入失败 {student_code}/{fname}: {e}")

    db.commit()
    cursor.close()
    db.close()

    print(f"\n✅ 迁移完成！")
    print(f"   总计扫描: {total} 张照片")
    print(f"   上传成功: {uploaded} 张")
    print(f"   已跳过:   {skipped} 张（已存在于 MinIO）")


if __name__ == "__main__":
    migrate()