-- teacher_long_memory 表结构升级
-- 执行前备份: SELECT * FROM teacher_long_memory;

ALTER TABLE teacher_long_memory
    ADD COLUMN focus_student_codes VARCHAR(255) DEFAULT '' COMMENT '常查询的学生学号',
    ADD COLUMN last_class_code VARCHAR(50) DEFAULT '' COMMENT '最近使用的班级',
    ADD COLUMN query_count INT DEFAULT 0 COMMENT '累计查询次数',
    ADD COLUMN last_query_time DATETIME DEFAULT NULL COMMENT '最近查询时间',
    ADD COLUMN prefer_focus_topic VARCHAR(100) DEFAULT '' COMMENT '关注话题偏好(行为分析/专注度/排行/人脸)',
    ADD COLUMN recent_queries TEXT DEFAULT '' COMMENT '最近5条问题(JSON数组)';