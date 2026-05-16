from datetime import datetime, timedelta
from dashscope import Generation
import dashscope
import json
import os
import pymysql
from minio import Minio

# ========================
# 配置
# ========================
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "user_center_db"),
    "charset": "utf8mb4"
}

MINIO_CLIENT = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    secure=False
)
BUCKET_NAME = "video-bucket"

# ========================
# 工具 1：获取老师历史报告
# ========================
def tool_get_teacher_all_reports(teacher_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT cr.id, cr.class_code, cr.lesson_section, cr.created_at, c.class_name
            FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code   # 修复！！！
            WHERE cr.teacher_code = %s
            ORDER BY cr.created_at DESC
        """, (teacher_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        if not rows:
            return f"老师 {teacher_code} 暂无课堂记录"

        res = [f"报告ID:{row['id']} | 班级:{row['class_name']} | 课程:{row['lesson_section']} | 时间:{row['created_at']}"
               for row in rows]
        return f"您的历史课堂（共{len(rows)}节）：\n" + "\n".join(res)
    except Exception as e:
        return f"获取历史失败：{str(e)}"

# ========================
# 工具 2：获取单节课详情
# ========================
def tool_get_single_report_detail(report_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        
        cursor.execute("""
            SELECT minio_json_path 
            FROM course_reports 
            WHERE report_code = %s
        """, (report_code,))
        
        report = cursor.fetchone()
        cursor.close()
        db.close()

        if not report or not report.get('minio_json_path'):
            return f"【报告 {report_code}】错误：未找到数据"

        # ========== 通用读取 JSON ==========
        try:
            resp = MINIO_CLIENT.get_object(BUCKET_NAME, report['minio_json_path'])
            json_data = resp.read()
            stats = json.loads(json_data)
            counts = stats.get("behavior_counts", {})
        except Exception as e:
            return f"【报告 {report_code}】读取失败：{str(e)}"

        # ========== 自动统计（不写死任何字段） ==========
        total = sum(counts.values())
        behavior_lines = [f"{name}：{num}次" for name, num in counts.items()]
        behavior_str = "\n".join(behavior_lines)

        # ========== 通用返回，不依赖标签 ==========
        return f"""
【报告 {report_code} 完整分析】
总行为次数：{total}次

行为明细：
{behavior_str}
"""
    except Exception as e:
        return f"【报告 {report_code}】异常：{str(e)}"

# ========================
# 工具 3：获取某班级所有学生列表 ✅ 连表查询版（完全匹配你的表）
# ========================
def tool_get_class_students(class_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT 
                s.student_code, 
                u.name, 
                s.gender, 
                s.age
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            WHERE s.class_code = %s 
            ORDER BY s.id
        """, (class_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        if not rows:
            return f"班级 {class_code} 暂无学生"

        lines = []
        for s in rows:
            lines.append(f"学号:{s['student_code']} | 姓名:{s['name']} | 性别:{s['gender']} | 年龄:{s['age']}")
        
        return f"班级 {class_code} 学生列表（共{len(rows)}人）：\n" + "\n".join(lines)
    except Exception as e:
        return f"获取学生失败：{str(e)}"

# ========================
# 工具 4：按时间段查询课堂统计 ✅ NEW
# ========================
def tool_get_time_range_stats(teacher_code: str, time_type: str = "7d"):
    try:
        now = datetime.now()
        if time_type == "7d":
            start = now - timedelta(days=7)
            title = "近7天"
        elif time_type == "30d" or time_type == "month":
            start = now - timedelta(days=30)
            title = "近30天"
        else:
            return "仅支持 7d / 30d"

        start_str = start.strftime("%Y-%m-%d 00:00:00")
        end_str = now.strftime("%Y-%m-%d %H:%M:%S")

        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT id, created_at, lesson_section, class_code 
            FROM course_reports
            WHERE teacher_code = %s AND created_at BETWEEN %s AND %s
            ORDER BY created_at
        """, (teacher_code, start_str, end_str))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        if not rows:
            return f"{title} 无课堂记录"

        res = [f"报告ID:{r['id']} | 节次:{r['lesson_section']} | 时间:{r['created_at']}" for r in rows]
        return f"📊 {title} 课堂统计（共{len(rows)}节）：\n" + "\n".join(res)
    except Exception as e:
        return f"统计失败：{str(e)}"
    
# ========================
# 工具 5：获取课堂关键帧
# ========================
def tool_get_class_keyframe(report_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT minio_keyframe_path
            FROM course_reports
            WHERE report_code = %s
        """, (report_code,))
        row = cursor.fetchone()
        cursor.close()
        db.close()

        if not row or not row['minio_keyframe_path']:
            return "该课堂暂无关键帧图片"

        url = MINIO_CLIENT.presigned_get_object(
            BUCKET_NAME,
            row['minio_keyframe_path'],
            expires=timedelta(days=7)
        )
        url = url.replace("http://minio:9000", "http://localhost:9000")
        return url

    except Exception as e:
        return f"获取关键帧失败：{str(e)}"
    
# ========================
# 工具 6：批量获取报告详情
# ========================
def tool_get_batch_report_detail(report_codes: list):
    """批量一次性获取多个报告详情"""
    res = []
    for rc in report_codes:
        detail = tool_get_single_report_detail(rc)
        res.append(detail)
    return str(res)

# ========================
# 工具 7：获取课程表
# ========================
def tool_get_teacher_schedule(teacher_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT week_day, section, class_name, course_name, classroom 
            FROM teacher_course_schedule 
            WHERE teacher_code=%s 
            ORDER BY week_day, section
        """, (teacher_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        if not rows:
            return "暂无课程安排"

        week_map = {1:"周一",2:"周二",3:"周三",4:"周四",5:"周五",6:"周六",7:"周日"}
        lines = ["📅 你的课程安排："]
        for r in rows:
            wd = week_map.get(r["week_day"], "未知")
            sec = r["section"]
            cls = r["class_name"]
            cou = r["course_name"]
            room = r["classroom"]
            lines.append(f"• {wd} 第{sec}节 | {cls} - {cou} | {room}")
        return "\n".join(lines)
    except Exception as e:
        return f"获取课表失败：{str(e)}"

# ========================
# 工具 8：查询班级人脸绑定状态
# ========================
def tool_get_face_mapping(class_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)

        # 1. 获取班级所有学生
        cursor.execute("""
            SELECT s.student_code, u.name
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            WHERE s.class_code = %s
        """, (class_code,))
        all_students = cursor.fetchall()

        # 2. 获取已绑定人脸
        cursor.execute("""
            SELECT face_id, student_code, student_name
            FROM face_student_mapping
            WHERE class_code = %s
        """, (class_code,))
        bound_rows = cursor.fetchall()
        cursor.close()
        db.close()

        bound_codes = {r["student_code"] for r in bound_rows if r.get("student_code")}
        bound_lines = []
        unbound_lines = []

        for s in all_students:
            if s["student_code"] in bound_codes:
                face_entry = next((r for r in bound_rows if r.get("student_code") == s["student_code"]), None)
                face_id = face_entry["face_id"] if face_entry else "未知"
                bound_lines.append(f"  ✅ {s['name']}（学号:{s['student_code']}）→ 人脸ID: {face_id}")
            else:
                unbound_lines.append(f"  ❌ {s['name']}（学号:{s['student_code']}）")

        result_parts = [f"班级 {class_code} 人脸绑定情况（共{len(all_students)}人）："]
        if bound_lines:
            result_parts.append(f"\n已绑定（{len(bound_lines)}人）：")
            result_parts.extend(bound_lines)
        if unbound_lines:
            result_parts.append(f"\n未绑定（{len(unbound_lines)}人）：")
            result_parts.extend(unbound_lines)
        if not bound_lines and not unbound_lines:
            result_parts.append("暂无学生数据")

        return "\n".join(result_parts)
    except Exception as e:
        return f"查询人脸绑定失败：{str(e)}"


# ========================
# 工具 9：班级专注度排行榜
# ========================
def tool_get_class_rank(class_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT 
                u.name AS name,
                s.student_code,
                IFNULL(AVG(r.focus_rate), 0) AS avg_focus,
                COUNT(r.id) AS report_count
            FROM students s
            JOIN users u ON s.user_code = u.user_code
            LEFT JOIN student_reports r 
                ON s.student_code = r.student_code 
                AND r.class_code = %s
            WHERE s.class_code = %s
            GROUP BY s.student_code, u.name
            HAVING avg_focus > 0
            ORDER BY avg_focus DESC
        """, (class_code, class_code))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        if not rows:
            return f"班级 {class_code} 暂无专注度数据"

        lines = [f"📊 班级 {class_code} 专注度排行榜（共{len(rows)}人）："]
        for i, r in enumerate(rows, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}."
            lines.append(
                f"  {medal} {r['name']}（{r['student_code']}）"
                f" — 平均专注度 {round(float(r['avg_focus']))}%"
                f"，共{r['report_count']}节课"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"查询排行榜失败：{str(e)}"


# ========================
# 工具 10：学生跨课堂行为汇总
# ========================
def tool_get_student_behavior(student_code: str, class_code: str):
    try:
        import json as _json
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)

        # 查该学生姓名
        cursor.execute("""
            SELECT u.name FROM students s
            JOIN users u ON s.user_code = u.user_code
            WHERE s.student_code = %s
        """, (student_code,))
        stu = cursor.fetchone()
        student_name = stu["name"] if stu else student_code

        # 查该班级所有课堂报告
        cursor.execute("""
            SELECT minio_json_path FROM course_reports
            WHERE class_code=%s ORDER BY created_at DESC
        """, (class_code,))
        reports = cursor.fetchall()
        cursor.close()
        db.close()

        if not reports:
            return f"学生 {student_name} 暂无课堂记录"

        total_behaviors = {}
        matched_count = 0
        for r in reports:
            try:
                data = MINIO_CLIENT.get_object(BUCKET_NAME, r["minio_json_path"])
                stats = _json.loads(data.read())
                sb = stats.get("student_behaviors", {})
                if student_code in sb:
                    matched_count += 1
                    for b, cnt in sb[student_code].items():
                        total_behaviors[b] = total_behaviors.get(b, 0) + cnt
            except:
                continue

        if not total_behaviors:
            return f"学生 {student_name}（{student_code}）在 {len(reports)} 节课中未被识别到行为数据（可能未绑定人脸）"

        lines = [f"📋 学生 {student_name}（{student_code}）行为汇总（在 {matched_count}/{len(reports)} 节课中被识别）："]
        total = sum(total_behaviors.values())
        for b, cnt in sorted(total_behaviors.items(), key=lambda x: -x[1]):
            pct = round(100 * cnt / total)
            lines.append(f"  • {b}：{cnt}次（{pct}%）")

        return "\n".join(lines)
    except Exception as e:
        return f"查询学生行为失败：{str(e)}"


# ========================
# 工具 11：获取报告AI分析全文
# ========================
def tool_get_report_ai_analysis(report_id: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT cr.id, cr.report_code, cr.minio_json_path, c.class_name, cr.lesson_section
            FROM course_reports cr
            JOIN classes c ON cr.class_code = c.class_code
            WHERE cr.id = %s OR cr.report_code = %s
        """, (report_id, report_id))
        report = cursor.fetchone()
        cursor.close()
        db.close()

        if not report:
            return f"报告 {report_id} 不存在"

        # 尝试读取 AI 分析
        ai_text = ""
        try:
            ai_path = report['minio_json_path'].replace("stats.json", "ai_report.md")
            ai_obj = MINIO_CLIENT.get_object(BUCKET_NAME, ai_path)
            ai_text = ai_obj.read().decode("utf-8")
        except:
            ai_text = ""

        if not ai_text:
            return f"报告 {report['report_code']}（{report['class_name']} - {report['lesson_section']}）暂无AI分析报告，请先在报告详情页生成。"

        return f"📝 报告 {report['report_code']}（{report['class_name']} - {report['lesson_section']}）AI分析：\n\n{ai_text}"
    except Exception as e:
        return f"获取AI分析失败：{str(e)}"


# ========================
# 工具定义
# ========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_get_teacher_all_reports",
            "description": "查询当前教师的所有历史课堂记录，教师工号系统已自动传入",
            "parameters": {
                "type": "object",
                "properties": {"teacher_code": {"type": "string"}},
                "required": ["teacher_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_single_report_detail",
            "description": "根据报告编号（如 R2025001）获取单节课专注度、分心等详细数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_code": {
                        "type": "string"
                    }
                },
                "required": ["report_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_class_students",
            "description": "根据班级ID获取该班级所有学生列表",
            "parameters": {
                "type": "object",
                "properties": {"class_code": {"type": "string"}},
                "required": ["class_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_time_range_stats",
            "description": "查询教师近7天/近30天的课堂统计，time_type 只能是 7d 或 30d",
            "parameters": {
                "type": "object",
                "properties": {
                    "teacher_code": {"type": "string"},
                    "time_type": {"type": "string"}
                },
                "required": ["teacher_code", "time_type"]
            }
        }
    },
    # 工具配置也同步改！
    {
        "type": "function",
        "function": {
            "name": "tool_get_class_keyframe",
            "description": "根据报告编号获取课堂关键帧图片链接",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "report_code": {
                            "type": "string"
                        }
                    },
                    "required": ["report_code"]
                }
        }
    },
{
    "type": "function",
    "function": {
        "name": "tool_get_batch_report_detail",
        "description": "批量一次性获取多个课堂报告的详细数据",
        "parameters": {
            "type": "object",
            "properties": {
                "report_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "报告代码列表，如['R2025001', 'R2025002']"
                }
            },
            "required": ["report_codes"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "tool_get_teacher_schedule",
        "description": "获取老师本周的课程安排、课表",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "tool_get_face_mapping",
        "description": "查询班级的人脸绑定情况，看哪些学生已绑定人脸、哪些未绑定",
        "parameters": {
            "type": "object",
            "properties": {
                "class_code": {"type": "string", "description": "班级编号"}
            },
            "required": ["class_code"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "tool_get_class_rank",
        "description": "查询班级学生专注度排行榜，按平均专注度排序",
        "parameters": {
            "type": "object",
            "properties": {
                "class_code": {"type": "string", "description": "班级编号"}
            },
            "required": ["class_code"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "tool_get_student_behavior",
        "description": "查询某个学生在所有课堂中的跨课堂行为汇总数据，需要学生学号和班级编号",
        "parameters": {
            "type": "object",
            "properties": {
                "student_code": {"type": "string", "description": "学生学号"},
                "class_code": {"type": "string", "description": "班级编号"}
            },
            "required": ["student_code", "class_code"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "tool_get_report_ai_analysis",
        "description": "获取某节课报告的AI分析全文，支持报告ID或报告编号",
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string", "description": "报告ID或报告编号（如 R2025001）"}
            },
            "required": ["report_id"]
        }
    }
}
]

# ========================
# MySQL 读写会话历史
# ========================
def get_session_messages(teacher_code, session_id):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT messages FROM chat_sessions
            WHERE teacher_code=%s AND session_id=%s
        """, (teacher_code, session_id))
        row = cursor.fetchone()
        cursor.close()
        db.close()
        if row:
            return json.loads(row["messages"])
    except:
        pass
    return []

def save_session_messages(teacher_code, session_id, title, messages):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        msg_json = json.dumps(messages, ensure_ascii=False)

        cursor.execute("""
            SELECT id FROM chat_sessions
            WHERE teacher_code=%s AND session_id=%s
        """, (teacher_code, session_id))
        exists = cursor.fetchone()

        if exists:
            cursor.execute("""
                UPDATE chat_sessions
                SET messages=%s, title=%s, update_time=NOW()
                WHERE teacher_code=%s AND session_id=%s
            """, (msg_json, title, teacher_code, session_id))
        else:
            cursor.execute("""
                INSERT INTO chat_sessions
                (teacher_code, session_id, title, messages)
                VALUES (%s, %s, %s, %s)
            """, (teacher_code, session_id, title, msg_json))
        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print("保存会话失败", e)

# 意图分类：固定5类，适配你整个系统
def detect_user_intent(question: str) -> str:
    """
    返回意图类型：
    query_history    - 查询个人历史课堂
    query_report      - 查询单节课详细报告/专注度/行为
    query_class       - 查询班级整体统计/纪律/人数/学生信息
    chat_chitchat     - 日常闲聊/问候
    other             - 无关问题、非业务请求
    """
    prompt = f"""
你是一个意图分类助手，请智能理解用户问题，只返回分类标识，不要输出任何多余内容。

可选标识（只能返回一个）：
query_history
query_report
query_class
chat_chitchat
other

【宽松判断规则】
1. query_history：只要涉及“我的课程、历史记录、上过的课、报告列表、有哪些课” → 归此类
2. query_report：只要涉及“某节课、专注度、分心、行为统计、报告详情、关键帧、图片” → 归此类
3. query_class：只要涉及“班级、学生、人数、名单、班级表现、纪律” → 归此类
4. chat_chitchat：问候、打招呼、介绍自己、闲聊、简单对话 → 归此类
5. other：与课堂分析、教学、学生、班级完全无关的内容 → 归此类

用户问题：{question}
请只返回标识：
    """

    resp = Generation.call(
        model='qwen-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        result_format='text',
        temperature=0.2  # 稍微提高一点，更灵活
    )
    intent = resp.output.text.strip()
    
    # 超级宽松容错：只要包含关键词就算对
    if "query_history" in intent:
        return "query_history"
    if "query_report" in intent:
        return "query_report"
    if "query_class" in intent:
        return "query_class"
    if "chat_chitchat" in intent:
        return "chat_chitchat"
    
    return "other"

def insert_agent_log(
    teacher_code,
    session_id,
    question,
    intent="",
    tool_calls="",
    tool_args="",
    tool_result="",
    final_answer=""
):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO agent_logs 
            (teacher_code, session_id, question, intent, tool_calls, tool_args, tool_result, final_answer)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            teacher_code,
            session_id,
            question,
            intent,
            tool_calls,
            tool_args,
            tool_result,
            final_answer
        ))
        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print("日志写入失败", e)

# ========================
# 核心对话接口
# ========================
def chat_agent_api(question: str, teacher_code: str, session_id: str):
    history = get_session_messages(teacher_code, session_id)

    try:
        # ========== 日志变量 ==========
        log_intent = ""
        log_tool_list = []
        log_args_list = []
        log_tool_res_list = []
        log_final_answer = ""
        
        # 1. 意图识别（闲聊拦截）
        intent = detect_user_intent(question)
        log_intent = intent
        
        if intent in ["chat_chitchat", "other"]:
            answer = "😊 我是课堂分析AI助手，我可以帮你：\n• 查询历史课堂记录与详情\n• 查询专注度/行为统计\n• 查询课堂关键帧画面\n• 多维度分析班级表现\n• 查询专注度排行榜\n• 查询人脸绑定情况\n• 查询单个学生跨课堂表现\n• 查看AI分析报告全文\n• 查询课程表安排" if intent=="chat_chitchat" else "🙏 抱歉，我只负责课堂行为分析相关问题。"
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            save_session_messages(teacher_code, session_id, question[:20] + "...", history)
            insert_agent_log(teacher_code, session_id, question, intent=log_intent, final_answer=answer)
            return {
                "answer": answer,
                "thinking_process": {
                    "intent": log_intent,
                    "tools": log_tool_list,
                    "args": log_args_list,
                    "results": log_tool_res_list
                }
            }

        # 2. 关键帧快捷识别
        q = question.lower()
        if any(key in q for key in ["关键帧", "keyframe", "图片", "照片"]):
            import re
            match = re.search(r'([Rr]\d+)', question, re.I)
            if match:
                report_code = match.group(1)  # 直接拿到 Rxxxx
                url = tool_get_class_keyframe(report_code)
                
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": url})
                
                save_session_messages(teacher_code, session_id, question[:20] + "...", history)
                insert_agent_log(
                            teacher_code, session_id, question,
                            intent=log_intent,
                            tool_calls="tool_get_class_keyframe",
                            tool_args=str({"report_code": report_code}),
                            tool_result=url,
                            final_answer=url
                        )                
                return url

        # 3. 判断对比问题
        question_clean = question.strip().lower()
        is_compare = any(w in question_clean for w in ["分心最严重","专注最高","最差","最好","对比","哪节课","谁最乱","谁最好","哪个课堂","哪个班","统计"])
        
        # 【增强】对比任务强制强提示+高温度
        if is_compare:
            base_prompt = """
你是专业课堂分析专家。
用户需要对比多节课数据时：
1. 必须先获取所有课堂报告ID
2. 必须获取所有报告详情后再给出结论
3. 禁止中途回复、禁止只查部分数据
4. 只输出最终结论，不要输出思考过程
5. 可以查询老师的课程表、本周课程、上课班级、上课教室
            """.strip()
            temperature=0.7
        else:
            base_prompt = "你是课堂分析助手，自主使用工具回答问题，数据不足继续调用，不编造、不输出中间思考。"
            temperature=0.1

        system_prompt = build_system_prompt_with_memory(teacher_code, base_prompt)

        # 【核心】独立推理链，不污染
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        max_round = 5
        current_round = 0
        answer = ""
        ai_msg = None

        while current_round < max_round:
            try:
                resp = dashscope.Generation.call(
                    model="qwen-plus",
                    messages=messages,
                    tools=TOOLS,
                    result_format="message",
                    temperature=temperature,
                    timeout=30
                )
            except Exception as e:
                current_round += 1
                continue

            choices = resp.output.get("choices", [])
            if not choices:
                answer = "数据分析失败"
                break

            ai_msg = choices[0]["message"]

            # --------------------------
            # 【关键修改】
            # 如果还在调用工具 → 绝对不保存中间文本！
            # --------------------------
            if "tool_calls" in ai_msg and ai_msg["tool_calls"]:
                tool = ai_msg["tool_calls"][0]
                func_name = tool["function"]["name"]
                args = json.loads(tool["function"]["arguments"])
                tool_result = ""

                log_tool_list.append(func_name)
                log_args_list.append(json.dumps(args))

                # ---- 分发工具调用 ----
                if func_name == "tool_get_teacher_all_reports":
                    tool_result = tool_get_teacher_all_reports(teacher_code)
                    update_teacher_long_memory(teacher_code, question=question, tool_name=func_name)
                elif func_name == "tool_get_single_report_detail":
                    tool_result = tool_get_single_report_detail(args.get("report_code"))
                    update_teacher_long_memory(teacher_code, report_code=args.get("report_code"), question=question, tool_name=func_name)
                elif func_name == "tool_get_batch_report_detail":
                    tool_result = tool_get_batch_report_detail(args.get("report_codes"))
                    update_teacher_long_memory(teacher_code, question=question, tool_name=func_name)
                elif func_name == "tool_get_class_students":
                    tool_result = tool_get_class_students(args.get("class_code"))
                    update_teacher_long_memory(teacher_code, class_code=args.get("class_code"), question=question, tool_name=func_name)
                elif func_name == "tool_get_time_range_stats":
                    tool_result = tool_get_time_range_stats(teacher_code, args.get("time_type", "7d"))
                    update_teacher_long_memory(teacher_code, question=question, tool_name=func_name)
                elif func_name == "tool_get_class_keyframe":
                    tool_result = tool_get_class_keyframe(args.get("report_code"))
                    update_teacher_long_memory(teacher_code, report_code=args.get("report_code"), question=question, tool_name=func_name)
                elif func_name == "tool_get_teacher_schedule":
                    tool_result = tool_get_teacher_schedule(teacher_code)
                    update_teacher_long_memory(teacher_code, question=question, tool_name=func_name)
                elif func_name == "tool_get_face_mapping":
                    tool_result = tool_get_face_mapping(args.get("class_code"))
                    update_teacher_long_memory(teacher_code, class_code=args.get("class_code"), question=question, tool_name=func_name)
                elif func_name == "tool_get_class_rank":
                    tool_result = tool_get_class_rank(args.get("class_code"))
                    update_teacher_long_memory(teacher_code, class_code=args.get("class_code"), question=question, tool_name=func_name)
                elif func_name == "tool_get_student_behavior":
                    tool_result = tool_get_student_behavior(
                        args.get("student_code"), args.get("class_code")
                    )
                    update_teacher_long_memory(teacher_code, class_code=args.get("class_code"), student_code=args.get("student_code"), question=question, tool_name=func_name)
                elif func_name == "tool_get_report_ai_analysis":
                    tool_result = tool_get_report_ai_analysis(args.get("report_id"))
                    update_teacher_long_memory(teacher_code, report_code=args.get("report_id"), question=question, tool_name=func_name)
                else:
                    tool_result = "未知工具"

                log_tool_res_list.append(tool_result)

                # 追加到推理链，但**绝对不返回给前端**
                messages.append(ai_msg)
                messages.append({"role": "tool", "content": tool_result, "name": func_name})
                current_round += 1
                continue

            # --------------------------
            # 只有结束工具调用，才取最终答案
            # --------------------------
            else:
                answer = ai_msg.get("content", "已完成分析").strip()
                break

        # 最终兜底
        if not answer:
            answer = ai_msg.get("content", "课堂分析完成") if ai_msg else "课堂分析完成"

        # 日志
        insert_agent_log(
            teacher_code, session_id, question,
            intent=log_intent,
            tool_calls=" | ".join(log_tool_list),
            tool_args=" | ".join(log_args_list),
            tool_result=" | ".join(log_tool_res_list),
            final_answer=answer
        )

        # 只写入最终结果
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        save_session_messages(teacher_code, session_id, question[:20] + "...", history)

        return {
            "answer": answer,
            "thinking_process": {
                "intent": log_intent,
                "tools": log_tool_list,
                "args": log_args_list,
                "results": log_tool_res_list
            }
        }

    except Exception as e:
        err = f"系统错误：{str(e)}"
        insert_agent_log(teacher_code, session_id, question, final_answer=err)
        # ========== 【修改】错误也返回统一格式 ==========
        return {
            "answer": err,
            "thinking_process": {
                "intent": "",
                "tools": [],
                "args": [],
                "results": []
            }
        }

def get_teacher_long_memory(teacher_code: str):
    """读取教师长期记忆（所有字段，缺失字段兜底空值）"""
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT * FROM teacher_long_memory WHERE teacher_code = %s
        """, (teacher_code,))
        row = cursor.fetchone()
        cursor.close()
        db.close()
        if not row:
            return {
                "focus_class_codes": "", "focus_report_codes": "", "prefer_question_type": "",
                "focus_student_codes": "", "last_class_code": "",
                "query_count": 0, "last_query_time": None,
                "prefer_focus_topic": "", "recent_queries": ""
            }
        # 兜底新字段（旧数据可能为 NULL）
        for key in ["focus_student_codes", "last_class_code", "prefer_focus_topic", "recent_queries"]:
            if not row.get(key):
                row[key] = ""
        if not row.get("query_count"):
            row["query_count"] = 0
        return row
    except:
        return {
            "focus_class_codes": "", "focus_report_codes": "", "prefer_question_type": "",
            "focus_student_codes": "", "last_class_code": "",
            "query_count": 0, "last_query_time": None,
            "prefer_focus_topic": "", "recent_queries": ""
        }


def update_teacher_long_memory(
    teacher_code: str,
    class_code=None, report_code=None, student_code=None,
    question=None, tool_name=None
):
    """
    每次工具调用后更新长期记忆，记录：
    - 常关注班级/报告/学生（去重，最多10个）
    - 最近使用的班级
    - 累计查询次数
    - 最近问题（滚动窗口5条）
    - 关注话题偏好（自动推断）
    """
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        old = get_teacher_long_memory(teacher_code)

        # ---- 追加班级 ----
        cls_list = [c for c in old["focus_class_codes"].split(",") if c] if old["focus_class_codes"] else []
        if class_code and str(class_code) not in cls_list:
            cls_list.append(str(class_code))
        new_cls = ",".join(cls_list[-10:])

        # ---- 追加报告 ----
        rep_list = [r for r in old["focus_report_codes"].split(",") if r] if old["focus_report_codes"] else []
        if report_code and str(report_code) not in rep_list:
            rep_list.append(str(report_code))
        new_rep = ",".join(rep_list[-10:])

        # ---- 追加学生 ----
        stu_list = [s for s in old["focus_student_codes"].split(",") if s] if old["focus_student_codes"] else []
        if student_code and str(student_code) not in stu_list:
            stu_list.append(str(student_code))
        new_stu = ",".join(stu_list[-10:])

        # ---- 最近使用班级 ----
        new_last_class = str(class_code) if class_code else old.get("last_class_code", "")

        # ---- 累计查询次数 ----
        new_count = (old.get("query_count") or 0) + 1

        # ---- 最近问题滚动窗口 ----
        try:
            recent = json.loads(old["recent_queries"]) if old.get("recent_queries") else []
        except:
            recent = []
        if question:
            recent.append(question[-80:])  # 截断防溢出
        recent = recent[-5:]  # 只保留最近5条
        new_recent = json.dumps(recent, ensure_ascii=False)

        # ---- 关注话题偏好 ----
        topic = old.get("prefer_focus_topic", "") or ""
        if tool_name:
            topic_map = {
                "tool_get_class_rank": "专注度排行",
                "tool_get_student_behavior": "学生行为分析",
                "tool_get_face_mapping": "人脸绑定管理",
                "tool_get_single_report_detail": "课堂行为报告",
                "tool_get_teacher_all_reports": "历史课堂查询",
                "tool_get_time_range_stats": "课堂趋势统计",
                "tool_get_report_ai_analysis": "AI分析报告",
                "tool_get_teacher_schedule": "课程表查询",
            }
            new_topic = topic_map.get(tool_name, topic)
        else:
            new_topic = topic

        # ---- 写入 ----
        cursor.execute("""
            INSERT INTO teacher_long_memory
            (teacher_code, focus_class_codes, focus_report_codes, prefer_question_type,
             focus_student_codes, last_class_code, query_count, last_query_time,
             prefer_focus_topic, recent_queries)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s)
            ON DUPLICATE KEY UPDATE
                focus_class_codes = VALUES(focus_class_codes),
                focus_report_codes = VALUES(focus_report_codes),
                focus_student_codes = VALUES(focus_student_codes),
                last_class_code = VALUES(last_class_code),
                query_count = VALUES(query_count),
                last_query_time = NOW(),
                prefer_focus_topic = VALUES(prefer_focus_topic),
                recent_queries = VALUES(recent_queries),
                update_time = NOW()
        """, (teacher_code, new_cls, new_rep, "",
              new_stu, new_last_class, new_count,
              new_topic, new_recent))

        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print("长期记忆更新失败", e)


def build_system_prompt_with_memory(teacher_code: str, base_prompt: str):
    mem = get_teacher_long_memory(teacher_code)
    recent_queries = ""
    try:
        rq = json.loads(mem["recent_queries"]) if mem.get("recent_queries") else []
        if rq:
            recent_queries = "、".join(f"「{q}」" for q in rq[-3:])
    except:
        pass

    extra = f"""
【用户长期习惯画像】
教师工号：{teacher_code}
累计互动：{mem.get('query_count', 0)} 次
常关注班级：{mem['focus_class_codes'] or '暂无'}
最近使用班级：{mem.get('last_class_code', '') or '暂无'}
常查看报告：{mem['focus_report_codes'] or '暂无'}
常查询学生：{mem.get('focus_student_codes', '') or '暂无'}
关注话题：{mem.get('prefer_focus_topic', '') or '暂无'}
最近提问：{recent_queries or '暂无'}

请结合以上习惯信息，优先使用该老师常用的班级和报告数据回答。
如果用户提到"我们班""我那个班"，优先使用「最近使用班级」的ID。
    """.strip()
    return base_prompt + "\n" + extra

# ========================
# 获取教师所有历史会话（给前端列表）
# ========================
def get_teacher_sessions(teacher_code):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT session_id, title, update_time
            FROM chat_sessions
            WHERE teacher_code=%s
            ORDER BY update_time DESC
        """, (teacher_code,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return rows
    except:
        return []
