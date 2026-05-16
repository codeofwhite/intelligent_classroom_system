-- ============================================
-- 智能课堂系统 数据库初始化脚本
-- 数据库: user_center_db
-- 共 13 张核心表
-- ============================================

CREATE DATABASE IF NOT EXISTS user_center_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE user_center_db;

-- ============================================
-- 1. 用户管理域（5张表）
-- ============================================

-- 用户基础表（所有角色共用）
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_code VARCHAR(50) NOT NULL UNIQUE COMMENT '用户唯一编码',
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '登录用户名',
    password VARCHAR(255) NOT NULL COMMENT '登录密码',
    name VARCHAR(50) NOT NULL COMMENT '用户姓名',
    role ENUM('student', 'parent', 'teacher') NOT NULL COMMENT '角色类型',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 班级表
CREATE TABLE IF NOT EXISTS classes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    class_code VARCHAR(50) NOT NULL UNIQUE COMMENT '班级编码',
    class_name VARCHAR(100) NOT NULL COMMENT '班级名称',
    grade VARCHAR(50) DEFAULT NULL COMMENT '年级',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 教师扩展表
CREATE TABLE IF NOT EXISTS teachers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_code VARCHAR(50) NOT NULL UNIQUE COMMENT '教师工号',
    user_code VARCHAR(50) NOT NULL COMMENT '关联用户编码',
    subject VARCHAR(100) DEFAULT NULL COMMENT '学科',
    class_code VARCHAR(50) DEFAULT NULL COMMENT '主管班级编码',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_code) REFERENCES users(user_code) ON DELETE CASCADE,
    FOREIGN KEY (class_code) REFERENCES classes(class_code) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 学生扩展表
CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_code VARCHAR(50) NOT NULL UNIQUE COMMENT '学号',
    user_code VARCHAR(50) NOT NULL COMMENT '关联用户编码',
    class_code VARCHAR(50) DEFAULT NULL COMMENT '所在班级编码',
    gender VARCHAR(10) DEFAULT NULL COMMENT '性别',
    age INT DEFAULT NULL COMMENT '年龄',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_code) REFERENCES users(user_code) ON DELETE CASCADE,
    FOREIGN KEY (class_code) REFERENCES classes(class_code) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 家长扩展表
CREATE TABLE IF NOT EXISTS parents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    parent_code VARCHAR(50) NOT NULL UNIQUE COMMENT '家长编码',
    user_code VARCHAR(50) NOT NULL COMMENT '关联用户编码',
    student_code VARCHAR(50) DEFAULT NULL COMMENT '关联学生学号',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_code) REFERENCES users(user_code) ON DELETE CASCADE,
    FOREIGN KEY (student_code) REFERENCES students(student_code) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- 2. 课堂报告域（2张表）
-- ============================================

-- 课堂报告主表
CREATE TABLE IF NOT EXISTS course_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    report_code VARCHAR(50) NOT NULL UNIQUE COMMENT '报告唯一编号',
    teacher_code VARCHAR(50) NOT NULL COMMENT '教师工号',
    class_code VARCHAR(50) NOT NULL COMMENT '班级编码',
    lesson_section VARCHAR(100) DEFAULT NULL COMMENT '课程节次/名称',
    minio_video_path VARCHAR(500) DEFAULT NULL COMMENT 'MinIO视频路径',
    minio_json_path VARCHAR(500) DEFAULT NULL COMMENT 'MinIO统计JSON路径',
    minio_csv_path VARCHAR(500) DEFAULT NULL COMMENT 'MinIO轨迹CSV路径',
    minio_keyframe_path VARCHAR(500) DEFAULT NULL COMMENT 'MinIO关键帧路径',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_teacher (teacher_code),
    INDEX idx_class (class_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 学生行为报告表
CREATE TABLE IF NOT EXISTS student_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_code VARCHAR(50) NOT NULL COMMENT '学生学号',
    class_code VARCHAR(50) DEFAULT NULL COMMENT '班级编码',
    report_code VARCHAR(50) DEFAULT NULL COMMENT '关联课堂报告编号',
    lesson_time DATETIME DEFAULT NULL COMMENT '上课时间',
    normal_posture INT DEFAULT 0 COMMENT '正常坐姿次数',
    raised_hand INT DEFAULT 0 COMMENT '举手次数',
    looking_down INT DEFAULT 0 COMMENT '低头次数',
    focus_rate FLOAT DEFAULT 0 COMMENT '专注度百分比',
    ai_comment TEXT DEFAULT NULL COMMENT 'AI评语',
    teacher_score INT DEFAULT NULL COMMENT '教师评分',
    teacher_comment TEXT DEFAULT NULL COMMENT '教师评语',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_student (student_code),
    INDEX idx_class (class_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- 3. 人脸签到域（2张表）
-- ============================================

-- 人脸图片表
CREATE TABLE IF NOT EXISTS face_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_code VARCHAR(50) NOT NULL COMMENT '学生学号',
    class_code VARCHAR(50) DEFAULT NULL COMMENT '班级编码',
    minio_path VARCHAR(500) NOT NULL COMMENT 'MinIO存储路径',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 人脸学生映射表
CREATE TABLE IF NOT EXISTS face_student_mapping (
    id INT AUTO_INCREMENT PRIMARY KEY,
    face_id VARCHAR(100) NOT NULL COMMENT '人脸标识',
    student_code VARCHAR(50) DEFAULT NULL COMMENT '学生学号',
    student_name VARCHAR(50) DEFAULT NULL COMMENT '学生姓名',
    class_code VARCHAR(50) DEFAULT NULL COMMENT '班级编码',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- 4. 课程排期域（1张表）
-- ============================================

-- 教师课程表
CREATE TABLE IF NOT EXISTS teacher_course_schedule (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_code VARCHAR(50) NOT NULL COMMENT '教师工号',
    week_day TINYINT NOT NULL COMMENT '星期几(1-7)',
    section VARCHAR(50) NOT NULL COMMENT '节次',
    class_name VARCHAR(100) DEFAULT NULL COMMENT '班级名称',
    course_name VARCHAR(100) DEFAULT NULL COMMENT '课程名称',
    classroom VARCHAR(100) DEFAULT NULL COMMENT '教室',
    INDEX idx_teacher (teacher_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- 5. AI智能域（3张表）
-- ============================================

-- AI对话会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_code VARCHAR(50) NOT NULL COMMENT '教师工号',
    session_id VARCHAR(100) NOT NULL COMMENT '会话唯一标识',
    title VARCHAR(200) DEFAULT NULL COMMENT '会话标题',
    messages JSON DEFAULT NULL COMMENT '完整对话历史(JSON数组)',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_session (teacher_code, session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Agent调用日志表
CREATE TABLE IF NOT EXISTS agent_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_code VARCHAR(50) DEFAULT NULL COMMENT '教师工号',
    session_id VARCHAR(100) DEFAULT NULL COMMENT '会话标识',
    question TEXT DEFAULT NULL COMMENT '用户问题',
    intent VARCHAR(50) DEFAULT NULL COMMENT '意图分类',
    tool_calls TEXT DEFAULT NULL COMMENT '调用的工具列表',
    tool_args TEXT DEFAULT NULL COMMENT '工具参数',
    tool_result TEXT DEFAULT NULL COMMENT '工具返回结果',
    final_answer TEXT DEFAULT NULL COMMENT '最终回答',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 教师长期记忆表
CREATE TABLE IF NOT EXISTS teacher_long_memory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_code VARCHAR(50) NOT NULL UNIQUE COMMENT '教师工号',
    focus_class_codes VARCHAR(500) DEFAULT NULL COMMENT '常关注班级',
    focus_report_codes VARCHAR(500) DEFAULT NULL COMMENT '常查看报告',
    focus_student_codes VARCHAR(500) DEFAULT NULL COMMENT '常查询学生',
    prefer_question_type VARCHAR(100) DEFAULT NULL COMMENT '偏好问题类型',
    last_class_code VARCHAR(50) DEFAULT NULL COMMENT '最近使用班级',
    query_count INT DEFAULT 0 COMMENT '累计查询次数',
    last_query_time DATETIME DEFAULT NULL COMMENT '最近查询时间',
    prefer_focus_topic VARCHAR(200) DEFAULT NULL COMMENT '关注话题偏好',
    recent_queries TEXT DEFAULT NULL COMMENT '最近问题(滚动窗口)',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- 测试数据
-- ============================================

-- 班级
INSERT INTO classes (class_code, class_name, grade) VALUES
('C001', '一年级一班', '一年级'),
('C002', '一年级二班', '一年级'),
('C003', '二年级一班', '二年级');

-- 用户
INSERT INTO users (user_code, username, password, name, role) VALUES
('U_T001', 'teacherwang', '123456', '王老师', 'teacher'),
('U_T002', 'teacherli',   '123456', '李老师', 'teacher'),
('U_S001', 'zhangsan',    '123456', '张三',   'student'),
('U_S002', 'lisi',        '123456', '李四',   'student'),
('U_S003', 'wangwu',      '123456', '王五',   'student'),
('U_P001', 'zhangfather', '123456', '张父',   'parent');

-- 教师
INSERT INTO teachers (teacher_code, user_code, subject, class_code) VALUES
('T2025001', 'U_T001', '数学', 'C001'),
('T2025002', 'U_T002', '语文', 'C002');

-- 学生
INSERT INTO students (student_code, user_code, class_code, gender, age) VALUES
('S2025001', 'U_S001', 'C001', '男', 7),
('S2025002', 'U_S002', 'C001', '女', 7),
('S2025003', 'U_S003', 'C002', '男', 8);

-- 家长
INSERT INTO parents (parent_code, user_code, student_code) VALUES
('P2025001', 'U_P001', 'S2025001');

-- 课程安排
INSERT INTO teacher_course_schedule (teacher_code, week_day, section, class_name, course_name, classroom) VALUES
('T2025001', 1, '第1-2节', '一年级一班', '数学', '教室101'),
('T2025001', 3, '第3-4节', '一年级一班', '数学', '教室101'),
('T2025001', 5, '第1-2节', '一年级二班', '数学', '教室102'),
('T2025002', 2, '第1-2节', '一年级二班', '语文', '教室102'),
('T2025002', 4, '第3-4节', '一年级二班', '语文', '教室102');