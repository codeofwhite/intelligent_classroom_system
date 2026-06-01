import pymysql
conn = pymysql.connect(host='localhost', port=3306, user='root', password='password123', database='user_center_db', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
c = conn.cursor()
c.execute("SELECT COUNT(*) as cnt FROM student_reports WHERE DATE(lesson_time)=CURDATE()")
print('Today reports:', c.fetchone()['cnt'])
c.execute("SELECT COUNT(*) as cnt FROM student_reports WHERE YEARWEEK(lesson_time)=YEARWEEK(NOW())")
print('This week reports:', c.fetchone()['cnt'])
c.execute("SELECT COUNT(*) as cnt FROM student_reports WHERE lesson_time >= DATE_FORMAT(NOW(), '%Y-%m-01')")
print('This month reports:', c.fetchone()['cnt'])
c.execute('SELECT student_code, focus_rate, behaviors_json FROM student_reports WHERE behaviors_json IS NOT NULL LIMIT 2')
for r in c.fetchall():
    print(f"  {r['student_code']}: focus={r['focus_rate']}, json={r['behaviors_json'][:100]}...")
conn.close()