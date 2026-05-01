from datetime import datetime, timedelta

import dashscope
import json
import pymysql
from minio import Minio

# ========================
# 配置
# ========================
dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password123",
    "database": "user_center_db",
    "charset": "utf8mb4"
}

MINIO_CLIENT = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
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
            SELECT cr.id, cr.class_id, cr.lesson_section, cr.created_at, c.class_name
            FROM course_reports cr
            JOIN classes c ON cr.class_id = c.id
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
def tool_get_single_report_detail(report_id: int):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT minio_json_path FROM course_reports WHERE id=%s", (report_id,))
        report = cursor.fetchone()
        cursor.close()
        db.close()

        if not report:
            return "未找到该报告"

        try:
            data = MINIO_CLIENT.get_object(BUCKET_NAME, report['minio_json_path'])
            stats = json.load(data.data)
            counts = stats["behavior_counts"]
        except:
            return f"报告ID {report_id}：已完成课堂分析，可查看专注度、分心行为、举手次数等数据。"

        total = sum(counts.values())
        focus = counts["举手"] + counts["看书"] + counts["写字"]
        distract = counts["使用手机"] + counts["低头做其他事情"] + counts["睡觉"]
        focus_rate = round(100 * focus / total, 1) if total > 0 else 0

        return f"""
【报告ID {report_id} 课堂详情】
专注率：{focus_rate}%
总行为次数：{total}
✅ 专注：举手{counts['举手']} 看书{counts['看书']} 写字{counts['写字']}
⚠️ 分心：手机{counts['使用手机']} 低头{counts['低头做其他事情']} 睡觉{counts['睡觉']}
"""
    except Exception as e:
        return f"获取课程详情失败：{str(e)}"

# ========================
# 工具 3：获取某班级所有学生列表 ✅ 连表查询版（完全匹配你的表）
# ========================
def tool_get_class_students(class_id: int):
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
            JOIN users u ON s.user_id = u.id
            WHERE s.class_id = %s 
            ORDER BY s.id
        """, (class_id,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        if not rows:
            return f"班级 {class_id} 暂无学生"

        lines = []
        for s in rows:
            lines.append(f"学号:{s['student_code']} | 姓名:{s['name']} | 性别:{s['gender']} | 年龄:{s['age']}")
        
        return f"班级 {class_id} 学生列表（共{len(rows)}人）：\n" + "\n".join(lines)
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
            SELECT id, created_at, lesson_section, class_id 
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
def tool_get_class_keyframe(report_id: int):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT minio_keyframe_path
            FROM course_reports
            WHERE id = %s
        """, (report_id,))
        row = cursor.fetchone()
        cursor.close()
        db.close()

        if not row or not row['minio_keyframe_path']:
            return "该课堂暂无关键帧图片"

        # 🔥 修复：过期时间用 timedelta，不是 int！
        from datetime import timedelta
        url = MINIO_CLIENT.presigned_get_object(
            BUCKET_NAME,
            row['minio_keyframe_path'],
            expires=timedelta(days=7)  # 这里改了！！！
        )

        url = url.replace("http://minio:9000", "http://localhost:9000")
        return url

    except Exception as e:
        return f"获取关键帧失败：{str(e)}"
    


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
            "description": "根据报告ID获取单节课专注度、分心等详细数据",
            "parameters": {
                "type": "object",
                "properties": {"report_id": {"type": "integer"}},
                "required": ["report_id"]
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
                "properties": {"class_id": {"type": "integer"}},
                "required": ["class_id"]
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
    {
        "type": "function",
        "function": {
            "name": "tool_get_class_keyframe",
            "description": "根据报告ID获取课堂关键帧图片链接",
            "parameters": {
                "type": "object",
                "properties": {"report_id": {"type": "integer"}},
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


# ========================
# 核心对话接口
# ========================
def chat_agent_api(question: str, teacher_code: str, session_id: str):
    history = get_session_messages(teacher_code, session_id)

    try:
        history.append({"role": "user", "content": question})

        # 强制关键词识别：只要提 keyframe / 关键帧 / 图片，直接返回
        q = question.lower()
        if any(key in q for key in ["关键帧", "keyframe", "图片", "照片"]):
            import re
            match = re.search(r'报告?[id\s:]*(\d+)', question, re.I)
            if match:
                report_id = int(match.group(1))
                url = tool_get_class_keyframe(report_id)
                history.append({"role": "assistant", "content": url})
                title = question[:20] + "..."
                save_session_messages(teacher_code, session_id, title, history)
                return url

        # 正常对话逻辑
        messages = [
            {
                "role": "system",
                "content": f"你是课堂分析AI助手，当前教师工号：{teacher_code}。直接调用工具，不要询问工号。"
            }
        ] + history[-6:]

        resp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=messages,
            tools=TOOLS,
            result_format="message"
        )

        output = resp.get("output", {})
        choices = output.get("choices", [])
        if not choices:
            return "AI 暂时无法回答"

        ai_msg = choices[0]["message"]

        if "tool_calls" in ai_msg and ai_msg["tool_calls"]:
            tool = ai_msg["tool_calls"][0]
            func_name = tool["function"]["name"]
            args = json.loads(tool["function"]["arguments"])

            if func_name == "tool_get_teacher_all_reports":
                tool_result = tool_get_teacher_all_reports(teacher_code)
            elif func_name == "tool_get_single_report_detail":
                tool_result = tool_get_single_report_detail(args.get("report_id"))
            elif func_name == "tool_get_class_students":
                tool_result = tool_get_class_students(args.get("class_id"))
            elif func_name == "tool_get_time_range_stats":
                tool_result = tool_get_time_range_stats(teacher_code, args.get("time_type", "7d"))
            elif func_name == "tool_get_class_keyframe":
                tool_result = tool_get_class_keyframe(args.get("report_id"))
            else:
                tool_result = "未知工具"

            messages.append(ai_msg)
            messages.append({
                "role": "tool",
                "content": tool_result,
                "name": func_name
            })

            final = dashscope.Generation.call(
                model="qwen-turbo",
                messages=messages,
                result_format="message"
            )
            answer = final["output"]["choices"][0]["message"]["content"]
        else:
            answer = ai_msg.get("content", "你好！我是课堂分析助手。")

        history.append({"role": "assistant", "content": answer})
        title = question[:20] + "..." if len(question) > 20 else question
        save_session_messages(teacher_code, session_id, title, history)

        return answer

    except Exception as e:
        return f"错误：{str(e)}"

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