-- init.sql
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password_hash VARCHAR(255), -- 预留密码位
    role ENUM('student', 'parent', 'teacher') NOT NULL,
    avatar_url VARCHAR(255),
    last_login_at DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (username, role) VALUES ('小探险家', 'student'), ('王妈妈', 'parent');