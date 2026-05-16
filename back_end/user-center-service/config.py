"""
配置模块 - 数据库连接配置
"""
import os


DB_CONFIG = {
    "host": os.getenv('DB_HOST', 'user-db'),
    "user": os.getenv('DB_USER', 'root'),
    "password": os.getenv('DB_PASSWORD', ''),
    "database": os.getenv('DB_NAME', 'user_center_db'),
    "port": int(os.getenv('DB_PORT', 3306)),
    "cursorclass": "DictCursor",  # 标记使用方式
}


def get_db_conn():
    """每次请求新建连接，用完关闭"""
    import pymysql
    return pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        port=DB_CONFIG["port"],
        cursorclass=pymysql.cursors.DictCursor,
    )