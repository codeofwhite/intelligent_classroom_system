from datetime import datetime, timedelta
from dashscope import Generation
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

# 意图分类：固定5类，适配你整个系统
def detect_user_intent(question: str) -> str:
    """
    返回意图类型：
    query_history    - 查询个人历史课堂
    query_report      - 查询单节课详细报告/专注度/行为
    query_class       - 查询班级整体统计/纪律
    chat_chitchat     - 日常闲聊/问候
    other             - 无关问题、非业务请求
    """
    prompt = f"""
请对用户问题做意图分类，只能严格返回下面其中一个标识，不要解释、不要多余文字：
可选标识：
query_history
query_report
query_class
chat_chitchat
other

规则：
1. 问自己上过的课堂、历史记录、有哪些分析报告 → query_history
2. 问某一节课专注度、课堂画面、行为统计、详细分析 → query_report
3. 问整个班级整体纪律、整体表现、班级统计 → query_class
4. 问候、打招呼、闲聊、你是谁、能干什么 → chat_chitchat
5. 写作文、娱乐、生活琐事、与课堂分析完全无关 → other

用户问题：{question}
    """

    resp = Generation.call(
        model='qwen-turbo',
        messages=[{'role':'user','content':prompt}],
        result_format='text',
        temperature=0.1
    )
    intent = resp.output.text.strip()
    # 兜底容错
    if intent not in ["query_history","query_report","query_class","chat_chitchat","other"]:
        return "other"
    return intent

# ========================
# 核心对话接口
# ========================
def chat_agent_api(question: str, teacher_code: str, session_id: str):
    history = get_session_messages(teacher_code, session_id)

    try:
        
        intent = detect_user_intent(question)
        
        if intent in ["chat_chitchat", "other"]:
            if intent == "chat_chitchat":
                answer = "😊 我是课堂分析AI助手，我可以帮你：\n• 查询历史课堂记录\n• 查询专注度/行为统计\n• 查看课堂关键帧画面\n• 分析班级表现"
            else:
                answer = "🙏 抱歉，我只负责课堂行为分析相关问题，无法回答这类内容哦~"
                
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            title = question[:20] + "..." if len(question) > 20 else question
            save_session_messages(teacher_code, session_id, title, history)
            return answer
        
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