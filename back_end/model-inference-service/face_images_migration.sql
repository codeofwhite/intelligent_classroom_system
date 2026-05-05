-- 人脸图片统一管理表
CREATE TABLE IF NOT EXISTS face_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_code VARCHAR(50) NOT NULL COMMENT '学生学号',
    class_code VARCHAR(50) NOT NULL COMMENT '班级编号',
    minio_path VARCHAR(255) NOT NULL COMMENT 'MinIO中的路径: {class_code}/{student_code}/xxx.jpg',
    created_at DATETIME DEFAULT NOW(),
    INDEX idx_student (student_code),
    INDEX idx_class (class_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人脸图片元数据（图片存MinIO face-images bucket）';