import pymysql
conn = pymysql.connect(host='localhost', port=3306, user='root', password='password123', database='user_center_db', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
c = conn.cursor()

# 先看当前数据
c.execute('SELECT id, report_code, lesson_time, created_at FROM course_reports ORDER BY id')
print("=== 当前 course_reports ===")
for r in c.fetchall():
    print(f"  ID:{r['id']} | {r['report_code']} | lesson_time:{r['lesson_time']} | created:{r['created_at']}")

# 更新时间 - 模拟不同日期的真实课堂
updates = [
    (1, '2025-10-13 08:00:00'),  # 周一 第1节 数学
    (2, '2025-10-27 08:00:00'),  # 周一 第1节 数学
    (3, '2025-11-10 08:00:00'),  # 周一 第1节 数学
    (4, '2025-10-14 10:00:00'),  # 周二 第3节 语文
    (5, '2025-10-28 14:00:00'),  # 周二 第5节 语文
]

for rid, dt in updates:
    c.execute("UPDATE course_reports SET lesson_time=%s, created_at=%s WHERE id=%s", (dt, dt, rid))

# 更新 lesson_section 为更合理的值
section_updates = [
    (1, '第1-2节'), (2, '第1-2节'), (3, '第1-2节'),
    (4, '第3-4节'), (5, '第5-6节'),
]
for rid, sec in section_updates:
    c.execute("UPDATE course_reports SET lesson_section=%s WHERE id=%s", (sec, rid))

conn.commit()

# 验证
c.execute('SELECT id, report_code, lesson_section, lesson_time FROM course_reports ORDER BY id')
print("\n=== 更新后 ===")
for r in c.fetchall():
    print(f"  ID:{r['id']} | {r['report_code']} | {r['lesson_section']} | {r['lesson_time']}")

cursor = c
cursor.close()
conn.close()
print("\nDone!")