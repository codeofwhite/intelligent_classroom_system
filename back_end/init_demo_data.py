"""
智能课堂系统 - 演示数据初始化脚本
用于生成拟真的班级数据，模拟一个真实小学四年级的教学场景
"""
import random
import json
import pymysql
from datetime import datetime, timedelta

# 数据库连接
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "password123",
    "database": "user_center_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

random.seed(42)  # 固定随机种子，保证可复现

# ============================================
# 1. 基础数据定义
# ============================================

CLASSES = [
    ("C001", "四年级一班", "四年级"),
    ("C002", "四年级二班", "四年级"),
]

TEACHERS = [
    # (teacher_code, user_code, username, name, subject, class_code)
    ("T2025001", "U_T001", "teacherwang", "王建国", "数学", "C001"),
    ("T2025002", "U_T002", "teacherli", "李芳", "语文", "C002"),
]

# 四年级一班 - 30名学生
CLASS1_STUDENTS = [
    "赵明轩", "钱思远", "孙浩然", "周子涵", "吴雨桐",
    "郑思琪", "冯嘉豪", "陈欣怡", "褚文博", "卫子墨",
    "蒋梦瑶", "沈宇航", "韩梓涵", "杨紫萱", "朱浩宇",
    "秦雅琪", "尤思远", "许子轩", "何雨欣", "吕佳琪",
    "施天佑", "张一诺", "孔德华", "曹博文", "严子琪",
    "华晨曦", "金浩轩", "魏子涵", "陶佳怡", "姜明辉",
]

# 四年级二班 - 28名学生
CLASS2_STUDENTS = [
    "谢雨彤", "邹思远", "柏子涵", "水梓萱", "窦浩然",
    "章梦琪", "云子轩", "苏欣然", "潘宇航", "葛雅琪",
    "奚子墨", "范雨欣", "彭嘉琪", "郎思琪", "鲁天佑",
    "韦博文", "昌一诺", "马德华", "苗佳怡", "凤明辉",
    "花晨曦", "俞浩轩", "任子涵", "袁雨桐", "柳雅婷",
    "酆子豪", "鲍思齐", "史嘉欣",
]


def generate_students(class_code, names, start_idx):
    """生成学生列表"""
    students = []
    for i, name in enumerate(names):
        idx = start_idx + i
        user_code = f"U_S{idx:03d}"
        student_code = f"S2025{idx:03d}"
        # 根据名字推断性别
        female_chars = set("萱瑶怡婷琪欣桐曦彤馨蕊妍")
        if any(c in female_chars for c in name):
            gender = "女"
        else:
            gender = random.choice(["男", "女"]) if random.random() > 0.3 else "男"
        students.append({
            "user_code": user_code,
            "username": f"stu{idx:03d}",
            "student_code": student_code,
            "name": name,
            "gender": gender,
            "age": random.choice([9, 10, 10, 10, 10, 11]),
            "class_code": class_code,
        })
    return students


def generate_parents(students, start_idx):
    """为每个学生生成一个家长"""
    parents = []
    for i, stu in enumerate(students):
        idx = start_idx + i
        suffix = random.choice(["爸爸", "妈妈"])
        parent_name = stu["name"][0] + suffix
        user_code = f"U_P{idx:03d}"
        parent_code = f"P2025{idx:03d}"
        parents.append({
            "user_code": user_code,
            "username": f"parent{idx:03d}",
            "parent_code": parent_code,
            "name": parent_name,
            "student_code": stu["student_code"],
        })
    return parents


# ============================================
# 2. 课程安排（section 为 int 类型，表示第几节）
# ============================================

SCHEDULES = [
    # 王建国老师 - 数学课
    # (teacher_code, week_day, section, class_name, course_name, classroom)
    ("T2025001", 1, 1, "四年级一班", "数学", "教室301"),
    ("T2025001", 2, 3, "四年级二班", "数学", "教室302"),
    ("T2025001", 3, 1, "四年级一班", "数学", "教室301"),
    ("T2025001", 4, 5, "四年级一班", "数学", "教室301"),
    ("T2025001", 5, 1, "四年级二班", "数学", "教室302"),

    # 李芳老师 - 语文课
    ("T2025002", 1, 3, "四年级二班", "语文", "教室302"),
    ("T2025002", 2, 1, "四年级一班", "语文", "教室301"),
    ("T2025002", 3, 5, "四年级二班", "语文", "教室302"),
    ("T2025002", 4, 1, "四年级二班", "语文", "教室302"),
    ("T2025002", 5, 3, "四年级一班", "语文", "教室301"),
]


# ============================================
# 3. 学生行为报告生成
# ============================================

AI_COMMENTS_HIGH = [
    "{name}同学在本节课中表现出色，全程保持高度专注，坐姿端正，积极举手发言。课堂参与度极高，是同学们学习的榜样。",
    "{name}同学上课认真听讲，注意力集中，举手发言踊跃。课上能紧跟老师节奏，学习态度端正，值得表扬。",
    "本节课{name}同学表现优秀，专注度持续稳定，能够主动参与课堂互动。坐姿良好，自律性强，继续保持！",
    "{name}同学今天课堂表现非常好，很少出现分心情况，对知识点的掌握也很扎实。期待你在后续课程中保持这个状态。",
]

AI_COMMENTS_MEDIUM = [
    "{name}同学本节课整体表现良好，大部分时间能保持专注，偶尔有低头走神的情况。建议加强自我约束，提升课堂效率。",
    "{name}同学上课时有一定的时间保持了注意力，但在后半节课出现了几次分心。建议课前做好预习，更容易跟上课堂节奏。",
    "本节课{name}同学的专注度处于中等水平，举手次数不多但坐姿基本端正。建议多参与课堂互动，积极回答问题。",
    "{name}同学今天课堂上偶尔会出现低头做其他事情的情况，不过大部分时间还是在认真听讲的。希望下次能更加专注。",
]

AI_COMMENTS_LOW = [
    "{name}同学本节课的课堂表现有待提升，低头次数较多，注意力不够集中。建议家长和老师共同关注，帮助孩子培养良好的学习习惯。",
    "{name}同学在课堂上多次出现分心行为，需要加强注意力管理。建议尝试分段式学习法，逐步提升专注时长。",
    "本节课{name}同学的课堂专注度较低，举手参与较少。建议老师在课上多给予关注和鼓励，帮助其建立学习信心。",
]

TEACHER_COMMENTS_HIGH = [
    "表现很棒，继续加油！",
    "课堂纪律好，学习态度认真。",
    "积极发言，思维活跃，非常好。",
    "专注度高，是班级的好榜样。",
]

TEACHER_COMMENTS_MEDIUM = [
    "还需更加专注，加油。",
    "偶尔走神，注意集中注意力。",
    "有进步，但举手可以更积极一些。",
    "课堂表现一般，需要努力。",
]

TEACHER_COMMENTS_LOW = [
    "上课注意力不够集中，需要改进。",
    "课堂纪律有待加强，请家长配合关注。",
    "建议课后多巩固，课堂上认真听讲。",
]


def generate_behavior_data(quality):
    """生成单节课的学生行为数据，返回 (normal, raised_hand, looking_down, focus_rate, behaviors_json)"""
    if quality == "high":
        normal = random.randint(18, 28)
        raised_hand = random.randint(3, 8)
        looking_down = random.randint(0, 3)
    elif quality == "medium":
        normal = random.randint(10, 18)
        raised_hand = random.randint(1, 4)
        looking_down = random.randint(3, 8)
    else:  # low
        normal = random.randint(4, 10)
        raised_hand = random.randint(0, 2)
        looking_down = random.randint(6, 15)

    total = normal + raised_hand + looking_down
    if total == 0:
        focus_rate = 100
    else:
        focus_rate = round((normal + raised_hand) / total * 100)

    # 构建 behaviors_json（兼容两种模型的标签）
    behaviors_json = json.dumps({
        "举手": raised_hand,
        "看书": max(0, normal - random.randint(0, 5)),
        "写字": random.randint(0, min(5, normal)),
        "使用手机": random.randint(0, max(0, looking_down - 3)),
        "低头做其他事情": max(0, looking_down - random.randint(0, 2)),
        "睡觉": random.randint(0, 1) if quality == "low" else 0,
        "正常坐姿": normal,
        "低头": looking_down
    }, ensure_ascii=False)

    return normal, raised_hand, looking_down, focus_rate, behaviors_json


def generate_ai_comment(name, quality):
    if quality == "high":
        return random.choice(AI_COMMENTS_HIGH).format(name=name)
    elif quality == "medium":
        return random.choice(AI_COMMENTS_MEDIUM).format(name=name)
    else:
        return random.choice(AI_COMMENTS_LOW).format(name=name)


def generate_teacher_score_and_comment(quality):
    if quality == "high":
        score = random.randint(85, 98)
        comment = random.choice(TEACHER_COMMENTS_HIGH)
    elif quality == "medium":
        score = random.randint(65, 84)
        comment = random.choice(TEACHER_COMMENTS_MEDIUM)
    else:
        score = random.randint(40, 64)
        comment = random.choice(TEACHER_COMMENTS_LOW)
    return score, comment


def generate_student_reports(students, class_code, teacher_code, lesson_dates):
    """为一组学生生成多个日期的报告"""
    reports = []

    # 给每个学生分配一个"表现等级"（有波动但有基础水平）
    student_quality = {}
    for s in students:
        r = random.random()
        if r < 0.25:
            student_quality[s["student_code"]] = "high"
        elif r < 0.75:
            student_quality[s["student_code"]] = "medium"
        else:
            student_quality[s["student_code"]] = "low"

    for lesson_date, lesson_section in lesson_dates:
        for stu in students:
            base_quality = student_quality[stu["student_code"]]
            # 有一定概率表现波动
            roll = random.random()
            if roll < 0.15:
                quality = {"low": "medium", "medium": "high", "high": "high"}[base_quality]
            elif roll < 0.25:
                quality = {"high": "medium", "medium": "low", "low": "low"}[base_quality]
            else:
                quality = base_quality

            normal, raised_hand, looking_down, focus_rate, behaviors_json = generate_behavior_data(quality)
            ai_comment = generate_ai_comment(stu["name"], quality)

            # 不是每个报告都有老师评分（有些老师还没评）
            has_teacher_score = random.random() < 0.7
            if has_teacher_score:
                teacher_score, teacher_comment = generate_teacher_score_and_comment(quality)
            else:
                teacher_score = None
                teacher_comment = None

            reports.append({
                "student_code": stu["student_code"],
                "class_code": class_code,
                "lesson_time": lesson_date + timedelta(hours=random.choice([8, 10, 14])),
                "normal_posture": normal,
                "raised_hand": raised_hand,
                "looking_down": looking_down,
                "behaviors_json": behaviors_json,
                "focus_rate": focus_rate,
                "ai_comment": ai_comment,
                "teacher_score": teacher_score,
                "teacher_comment": teacher_comment,
            })

    return reports


# ============================================
# 4. 人脸数据
# ============================================

def generate_face_data(students, class_code, start_idx):
    """生成人脸相关数据"""
    face_images = []
    face_mappings = []

    for i, stu in enumerate(students):
        idx = start_idx + i
        face_id = f"face_{stu['student_code']}"

        face_mappings.append({
            "face_id": face_id,
            "student_code": stu["student_code"],
            "student_name": stu["name"],
            "class_code": class_code,
        })

        # 每人1-3张照片
        num_photos = random.randint(1, 3)
        for j in range(num_photos):
            face_images.append({
                "student_code": stu["student_code"],
                "class_code": class_code,
                "minio_path": f"face_images/{stu['student_code']}/{j+1}.jpg",
            })

    return face_images, face_mappings


# ============================================
# 5. AI对话数据
# ============================================

def generate_chat_data(teacher_code):
    """为教师生成AI对话会话"""
    sessions = []
    logs = []

    # 会话1：班级整体分析
    session1_id = f"session_{teacher_code}_001"
    session1_messages = [
        {"role": "user", "content": "帮我分析一下四年级一班最近一周的整体课堂表现"},
        {"role": "assistant", "content": "好的，我来为您分析四年级一班最近一周的课堂表现数据。\n\n📊 **整体概况**：\n- 本周共进行了5节课，全班平均专注度为 **82.3%**\n- 较上周提升了 **2.1%**，整体呈上升趋势\n\n📈 **分项分析**：\n- 正常坐姿平均次数：18.5次/节课\n- 举手发言平均：3.2次/节课\n- 低头分心平均：4.1次/节课\n\n🏆 **表现优秀学生**：赵明轩、钱思远、孙浩然，专注度均超过90%\n⚠️ **需要关注学生**：严子琪、华晨曦，专注度低于65%\n\n建议对表现优秀的学生给予表扬，对需要关注的学生进行一对一沟通。"},
    ]
    sessions.append({
        "teacher_code": teacher_code,
        "session_id": session1_id,
        "title": "四年级一班课堂表现分析",
        "messages": json.dumps(session1_messages, ensure_ascii=False),
    })
    logs.append({
        "teacher_code": teacher_code,
        "session_id": session1_id,
        "question": "帮我分析一下四年级一班最近一周的整体课堂表现",
        "intent": "class_analysis",
        "tool_calls": "get_class_stats,get_student_rank",
        "tool_args": json.dumps({"class_code": "C001", "period": "week"}),
        "tool_result": json.dumps({"avg_focus": 82.3, "trend": "+2.1%"}),
        "final_answer": "四年级一班本周平均专注度82.3%，较上周提升2.1%。",
    })

    # 会话2：个别学生分析
    session2_id = f"session_{teacher_code}_002"
    session2_messages = [
        {"role": "user", "content": "帮我看看赵明轩同学这个月的表现趋势"},
        {"role": "assistant", "content": "赵明轩同学本月表现分析：\n\n📊 **月度数据**：\n- 平均专注度：**93.5%**\n- 共上课12节，其中10节专注度超过90%\n- 举手发言次数总计：42次（平均3.5次/节）\n\n📈 **趋势**：\n整体表现稳定且呈小幅上升趋势，上周有2节课专注度略降至85%，但本周已恢复到95%以上。\n\n✅ **评价**：该同学学习态度端正，自律性强，是班级的优秀榜样。建议适当给予更多挑战性任务，激发其潜力。"},
        {"role": "user", "content": "那严子琪呢？"},
        {"role": "assistant", "content": "严子琪同学本月表现分析：\n\n📊 **月度数据**：\n- 平均专注度：**58.2%**\n- 共上课12节，仅有3节专注度超过70%\n- 低头次数总计：89次（平均7.4次/节）\n- 举手次数：6次（平均0.5次/节）\n\n⚠️ **问题**：\n该同学课堂注意力持续性较差，尤其在第3-4节课后半段容易出现分心。可能与学习内容难度或个人状态有关。\n\n💡 **建议**：\n1. 与家长沟通了解近期生活状态\n2. 课堂上适当增加互动和关注\n3. 安排与专注度高的同学同桌\n4. 尝试分解任务，降低学习压力"},
    ]
    sessions.append({
        "teacher_code": teacher_code,
        "session_id": session2_id,
        "title": "学生个别表现分析",
        "messages": json.dumps(session2_messages, ensure_ascii=False),
    })
    logs.append({
        "teacher_code": teacher_code,
        "session_id": session2_id,
        "question": "帮我看看赵明轩同学这个月的表现趋势",
        "intent": "student_analysis",
        "tool_calls": "get_student_stats,get_student_trend",
        "tool_args": json.dumps({"student_code": "S2025001", "period": "month"}),
        "tool_result": json.dumps({"avg_focus": 93.5, "total_classes": 12}),
        "final_answer": "赵明轩本月平均专注度93.5%，表现优秀且稳定。",
    })
    logs.append({
        "teacher_code": teacher_code,
        "session_id": session2_id,
        "question": "那严子琪呢？",
        "intent": "student_analysis",
        "tool_calls": "get_student_stats,get_student_trend",
        "tool_args": json.dumps({"student_code": "S2025025", "period": "month"}),
        "tool_result": json.dumps({"avg_focus": 58.2, "total_classes": 12}),
        "final_answer": "严子琪本月平均专注度58.2%，需要重点关注。",
    })

    # 会话3：课程对比
    session3_id = f"session_{teacher_code}_003"
    session3_messages = [
        {"role": "user", "content": "对比一下我这周数学课和语文课的课堂效果"},
        {"role": "assistant", "content": "为您对比本周两门课程的课堂效果：\n\n📚 **数学课**（您执教）：\n- 平均专注度：**85.7%**\n- 平均举手次数：3.8次/节\n- 平均低头次数：3.2次/节\n\n📖 **语文课**（李芳老师执教）：\n- 平均专注度：**79.4%**\n- 平均举手次数：2.6次/节\n- 平均低头次数：5.1次/节\n\n📊 **分析**：\n数学课的整体表现优于语文课，可能与数学课互动性更强有关。建议语文课增加趣味性互动环节，提升学生参与度。"},
    ]
    sessions.append({
        "teacher_code": teacher_code,
        "session_id": session3_id,
        "title": "课程效果对比分析",
        "messages": json.dumps(session3_messages, ensure_ascii=False),
    })
    logs.append({
        "teacher_code": teacher_code,
        "session_id": session3_id,
        "question": "对比一下我这周数学课和语文课的课堂效果",
        "intent": "course_comparison",
        "tool_calls": "get_course_stats,compare_courses",
        "tool_args": json.dumps({"teacher_code": teacher_code, "period": "week"}),
        "tool_result": json.dumps({"math_avg": 85.7, "chinese_avg": 79.4}),
        "final_answer": "数学课平均专注度85.7%，语文课79.4%，数学课表现更优。",
    })

    return sessions, logs


def generate_teacher_memory(teacher_code, class_codes, student_codes):
    """生成教师长期记忆"""
    return {
        "teacher_code": teacher_code,
        "focus_class_codes": ",".join(class_codes),
        "focus_report_codes": "",
        "focus_student_codes": ",".join(student_codes[:5]),
        "prefer_question_type": "学生分析,班级统计",
        "last_class_code": class_codes[0],
        "query_count": random.randint(15, 50),
        "last_query_time": datetime.now(),
        "prefer_focus_topic": "专注度趋势,学生表现对比",
        "recent_queries": json.dumps([
            "班级整体表现如何",
            "哪些学生需要关注",
            "本周专注度趋势",
        ], ensure_ascii=False),
    }


# ============================================
# 6. 课堂报告主表
# ============================================

# 课程节次名称映射
SECTION_MAP = {1: "第1节", 2: "第2节", 3: "第3节", 4: "第4节", 5: "第5节", 6: "第6节"}


def generate_course_reports(lesson_dates_class1, lesson_dates_class2):
    """生成课堂报告主表记录"""
    reports = []

    # 从 C001 的课程中选3个日期创建课堂报告
    for i, (date, section_num, subject) in enumerate(lesson_dates_class1[:3]):
        report_code = f"CR_C001_{date.strftime('%Y%m%d')}_{i+1:02d}"
        reports.append({
            "report_code": report_code,
            "teacher_code": "T2025001",
            "class_code": "C001",
            "subject": subject,
            "lesson_time": date + timedelta(hours=8),
            "lesson_section": SECTION_MAP.get(section_num, f"第{section_num}节"),
            "minio_video_path": f"videos/C001/{report_code}.mp4",
            "minio_json_path": f"reports/C001/{report_code}/stats.json",
            "minio_csv_path": f"reports/C001/{report_code}/tracks.csv",
            "minio_keyframe_path": f"reports/C001/{report_code}/keyframe.jpg",
        })

    # 从 C002 的课程中选2个日期创建课堂报告
    for i, (date, section_num, subject) in enumerate(lesson_dates_class2[:2]):
        report_code = f"CR_C002_{date.strftime('%Y%m%d')}_{i+1:02d}"
        reports.append({
            "report_code": report_code,
            "teacher_code": "T2025002",
            "class_code": "C002",
            "subject": subject,
            "lesson_time": date + timedelta(hours=10),
            "lesson_section": SECTION_MAP.get(section_num, f"第{section_num}节"),
            "minio_video_path": f"videos/C002/{report_code}.mp4",
            "minio_json_path": f"reports/C002/{report_code}/stats.json",
            "minio_csv_path": f"reports/C002/{report_code}/tracks.csv",
            "minio_keyframe_path": f"reports/C002/{report_code}/keyframe.jpg",
        })

    return reports


# ============================================
# 主函数
# ============================================

def main():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("=" * 50)
    print("开始初始化演示数据...")
    print("=" * 50)

    # ---- 1. 班级 ----
    print("\n[1/8] 插入班级数据...")
    for class_code, class_name, grade in CLASSES:
        cursor.execute(
            "INSERT INTO classes (class_code, class_name, grade) VALUES (%s, %s, %s)",
            (class_code, class_name, grade)
        )
    print(f"  ✓ 插入 {len(CLASSES)} 个班级")

    # ---- 2. 用户 + 教师 ----
    print("\n[2/8] 插入教师数据...")
    for tc, uc, uname, name, subject, cc in TEACHERS:
        cursor.execute(
            "INSERT INTO users (user_code, username, password, name, role) VALUES (%s, %s, %s, %s, 'teacher')",
            (uc, uname, "123456", name)
        )
        cursor.execute(
            "INSERT INTO teachers (teacher_code, user_code, subject, class_code) VALUES (%s, %s, %s, %s)",
            (tc, uc, subject, cc)
        )
    print(f"  ✓ 插入 {len(TEACHERS)} 位教师")

    # ---- 3. 学生 ----
    print("\n[3/8] 插入学生数据...")
    class1_students = generate_students("C001", CLASS1_STUDENTS, 1)
    class2_students = generate_students("C002", CLASS2_STUDENTS, 31)
    all_students = class1_students + class2_students

    for stu in all_students:
        cursor.execute(
            "INSERT INTO users (user_code, username, password, name, role) VALUES (%s, %s, %s, %s, 'student')",
            (stu["user_code"], stu["username"], "123456", stu["name"])
        )
        cursor.execute(
            "INSERT INTO students (student_code, user_code, class_code, gender, age) VALUES (%s, %s, %s, %s, %s)",
            (stu["student_code"], stu["user_code"], stu["class_code"], stu["gender"], stu["age"])
        )
    print(f"  ✓ 插入 {len(all_students)} 名学生")

    # ---- 4. 家长 ----
    print("\n[4/8] 插入家长数据...")
    class1_parents = generate_parents(class1_students, 1)
    class2_parents = generate_parents(class2_students, 31)
    all_parents = class1_parents + class2_parents

    for par in all_parents:
        cursor.execute(
            "INSERT INTO users (user_code, username, password, name, role) VALUES (%s, %s, %s, %s, 'parent')",
            (par["user_code"], par["username"], "123456", par["name"])
        )
        cursor.execute(
            "INSERT INTO parents (parent_code, user_code, student_code) VALUES (%s, %s, %s)",
            (par["parent_code"], par["user_code"], par["student_code"])
        )
    print(f"  ✓ 插入 {len(all_parents)} 位家长")

    # ---- 5. 课程表（section 为 int） ----
    print("\n[5/8] 插入课程安排...")
    for tc, wd, sec, cn, crs, room in SCHEDULES:
        cursor.execute(
            "INSERT INTO teacher_course_schedule (teacher_code, week_day, section, class_name, course_name, classroom) VALUES (%s, %s, %s, %s, %s, %s)",
            (tc, wd, sec, cn, crs, room)
        )
    print(f"  ✓ 插入 {len(SCHEDULES)} 条课程安排")

    # ---- 6. 学生报告 ----
    print("\n[6/8] 生成学生课堂行为报告...")

    # C001 过去12周的上课日期（周一、周三、周四）
    base_date = datetime(2025, 9, 8)  # 2025年9月8日（周一）
    lesson_dates_c1 = []
    for week in range(12):
        offset = timedelta(weeks=week)
        lesson_dates_c1.append((base_date + offset + timedelta(days=0), "第1-2节"))      # 周一
        lesson_dates_c1.append((base_date + offset + timedelta(days=2), "第1-2节"))      # 周三
        lesson_dates_c1.append((base_date + offset + timedelta(days=3), "第5-6节"))      # 周四

    # C002 过去12周的上课日期（周一、周二、周五）
    lesson_dates_c2 = []
    for week in range(12):
        offset = timedelta(weeks=week)
        lesson_dates_c2.append((base_date + offset + timedelta(days=0), "第3-4节"))      # 周一
        lesson_dates_c2.append((base_date + offset + timedelta(days=1), "第1-2节"))      # 周二
        lesson_dates_c2.append((base_date + offset + timedelta(days=4), "第3-4节"))      # 周五

    # ✅ 补充近期数据（本周 + 本月 + 今日），确保 Medal 页面统计数据有值
    today = datetime.now()
    # 本周一
    this_monday = today - timedelta(days=today.weekday())
    # 本月1号
    this_month_1 = today.replace(day=1)

    # 补充本周数据（周一、周三如果有课的话）
    for day_offset in [0, 2, 3]:  # 周一、周三、周四
        lesson_day = this_monday + timedelta(days=day_offset)
        if lesson_day.date() <= today.date():
            lesson_dates_c1.append((lesson_day, "第1-2节"))

    for day_offset in [0, 1, 4]:  # 周一、周二、周五
        lesson_day = this_monday + timedelta(days=day_offset)
        if lesson_day.date() <= today.date():
            lesson_dates_c2.append((lesson_day, "第3-4节"))

    # 补充上个月数据（确保本月统计不为空）
    last_month_start = (this_month_1 - timedelta(days=1)).replace(day=1)
    for week in range(3):
        offset = timedelta(weeks=week)
        lesson_dates_c1.append((last_month_start + timedelta(days=0) + offset, "第1-2节"))
        lesson_dates_c2.append((last_month_start + timedelta(days=1) + offset, "第1-2节"))

    reports_c1 = generate_student_reports(class1_students, "C001", "T2025001", lesson_dates_c1)
    reports_c2 = generate_student_reports(class2_students, "C002", "T2025002", lesson_dates_c2)
    all_reports = reports_c1 + reports_c2

    # 批量插入（含 behaviors_json）
    insert_sql = """
        INSERT INTO student_reports
        (student_code, class_code, lesson_time, normal_posture, raised_hand, looking_down, behaviors_json, focus_rate, ai_comment, teacher_score, teacher_comment)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    batch_size = 500
    for i in range(0, len(all_reports), batch_size):
        batch = all_reports[i:i+batch_size]
        cursor.executemany(insert_sql, [
            (r["student_code"], r["class_code"], r["lesson_time"],
             r["normal_posture"], r["raised_hand"], r["looking_down"], r["behaviors_json"], r["focus_rate"],
             r["ai_comment"], r["teacher_score"], r["teacher_comment"])
            for r in batch
        ])
    print(f"  ✓ 插入 {len(all_reports)} 条学生报告（C001: {len(reports_c1)}, C002: {len(reports_c2)}）")

    # ---- 7. 课堂报告主表 ----
    print("\n[7/8] 生成课堂报告主表...")

    # 构建课表报告的日期数据（带 subject 信息）
    lesson_dates_c1_for_cr = [
        (base_date + timedelta(weeks=0), 1, "数学"),
        (base_date + timedelta(weeks=2), 1, "数学"),
        (base_date + timedelta(weeks=4), 1, "数学"),
    ]
    lesson_dates_c2_for_cr = [
        (base_date + timedelta(weeks=0), 3, "语文"),
        (base_date + timedelta(weeks=2), 5, "语文"),
    ]

    course_reports = generate_course_reports(lesson_dates_c1_for_cr, lesson_dates_c2_for_cr)
    for cr in course_reports:
        cursor.execute(
            """INSERT INTO course_reports
            (report_code, teacher_code, class_code, subject, lesson_time, lesson_section, minio_video_path, minio_json_path, minio_csv_path, minio_keyframe_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (cr["report_code"], cr["teacher_code"], cr["class_code"], cr["subject"],
             cr["lesson_time"], cr["lesson_section"], cr["minio_video_path"],
             cr["minio_json_path"], cr["minio_csv_path"], cr["minio_keyframe_path"])
        )
    print(f"  ✓ 插入 {len(course_reports)} 条课堂报告")

    # ---- 8. 人脸数据 ----
    print("\n[8/8] 生成人脸签到数据...")
    face_imgs_c1, face_maps_c1 = generate_face_data(class1_students, "C001", 1)
    face_imgs_c2, face_maps_c2 = generate_face_data(class2_students, "C002", 31)
    all_face_imgs = face_imgs_c1 + face_imgs_c2
    all_face_maps = face_maps_c1 + face_maps_c2

    for fi in all_face_imgs:
        cursor.execute(
            "INSERT INTO face_images (student_code, class_code, minio_path) VALUES (%s, %s, %s)",
            (fi["student_code"], fi["class_code"], fi["minio_path"])
        )
    for fm in all_face_maps:
        cursor.execute(
            "INSERT INTO face_student_mapping (face_id, student_code, student_name, class_code) VALUES (%s, %s, %s, %s)",
            (fm["face_id"], fm["student_code"], fm["student_name"], fm["class_code"])
        )
    print(f"  ✓ 插入 {len(all_face_imgs)} 条人脸图片记录")
    print(f"  ✓ 插入 {len(all_face_maps)} 条人脸学生映射")

    # ---- 9. AI对话数据 ----
    print("\n[附加] 生成AI对话数据...")
    for tc, uc, uname, name, subject, cc in TEACHERS:
        sessions, logs = generate_chat_data(tc)

        for s in sessions:
            cursor.execute(
                "INSERT INTO chat_sessions (teacher_code, session_id, title, messages) VALUES (%s, %s, %s, %s)",
                (s["teacher_code"], s["session_id"], s["title"], s["messages"])
            )

        for l in logs:
            cursor.execute(
                """INSERT INTO agent_logs
                (teacher_code, session_id, question, intent, tool_calls, tool_args, tool_result, final_answer)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (l["teacher_code"], l["session_id"], l["question"], l["intent"],
                 l["tool_calls"], l["tool_args"], l["tool_result"], l["final_answer"])
            )

        # 教师长期记忆
        class_codes = [cc]
        student_codes = [s["student_code"] for s in (class1_students if cc == "C001" else class2_students)]
        mem = generate_teacher_memory(tc, class_codes, student_codes)
        cursor.execute(
            """INSERT INTO teacher_long_memory
            (teacher_code, focus_class_codes, focus_report_codes, focus_student_codes,
             prefer_question_type, last_class_code, query_count, last_query_time,
             prefer_focus_topic, recent_queries)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (mem["teacher_code"], mem["focus_class_codes"], mem["focus_report_codes"],
             mem["focus_student_codes"], mem["prefer_question_type"], mem["last_class_code"],
             mem["query_count"], mem["last_query_time"], mem["prefer_focus_topic"],
             mem["recent_queries"])
        )

    print(f"  ✓ 插入 {len(TEACHERS) * 3} 个AI会话, {len(TEACHERS) * 4} 条Agent日志, {len(TEACHERS)} 条教师记忆")

    # 提交
    conn.commit()
    cursor.close()
    conn.close()

    print("\n" + "=" * 50)
    print("✅ 所有演示数据初始化完成！")
    print("=" * 50)
    print(f"\n📊 数据统计:")
    print(f"  班级: 2 个")
    print(f"  教师: 2 位")
    print(f"  学生: {len(all_students)} 名")
    print(f"  家长: {len(all_parents)} 位")
    print(f"  课程安排: {len(SCHEDULES)} 条")
    print(f"  学生报告: {len(all_reports)} 条")
    print(f"  课堂报告: {len(course_reports)} 条")
    print(f"  人脸图片: {len(all_face_imgs)} 条")
    print(f"  人脸映射: {len(all_face_maps)} 条")

    print(f"\n🔑 测试账号:")
    print(f"  教师1: teacherwang / 123456 (王建国 - 数学)")
    print(f"  教师2: teacherli / 123456 (李芳 - 语文)")
    print(f"  学生1: stu001 / 123456 (赵明轩)")
    print(f"  家长1: parent001 / 123456 (赵爸爸/赵妈妈)")


if __name__ == "__main__":
    main()