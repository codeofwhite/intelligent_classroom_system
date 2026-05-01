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

# ========================
# 核心对话接口
# ========================
def chat_agent_api(question: str, teacher_code: str, session_id: str):
    history = get_session_messages(teacher_code, session_id)

    try:
        # 1. 意图识别（闲聊拦截）
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

        # 2. 关键帧快捷识别
        q = question.lower()
        if any(key in q for key in ["关键帧", "keyframe", "图片", "照片"]):
            import re
            match = re.search(r'报告?[id\s:]*(\d+)', question, re.I)
            if match:
                report_id = int(match.group(1))
                url = tool_get_class_keyframe(report_id)
                history.append({"role": "assistant", "content": url})
                save_session_messages(teacher_code, session_id, question[:20] + "...", history)
                return url

        # ==============================
        # 多轮反思 + 链式调用 增强版
        # ==============================
        base_prompt  = f"""
你是课堂分析AI助手，教师工号：{teacher_code}。
重要规则：
1. 用户问「哪节课分心最严重、哪节专注率最高」这类对比问题：
   第一步先调用 tool_get_teacher_all_reports 获取所有报告ID
   第二步再逐个调用 tool_get_single_report_detail 查询每节课详情
   第三步自己对比专注率、分心情况，给出明确结论
2. 允许连续多轮工具调用，不要中途放弃
3. 直到收集完足够数据、能给出明确对比答案再结束
4. 不要让用户说得更具体，你自己主动分步查询补齐数据
        """.strip()
        system_prompt = build_system_prompt_with_memory(teacher_code, base_prompt)
        messages = [{"role": "system", "content": system_prompt}] + history[-6:]

        max_round = 5   # 放宽到5轮，足够查列表+多节课详情
        current_round = 0
        answer = ""

        while current_round < max_round:
            resp = dashscope.Generation.call(
                model="qwen-turbo",
                messages=messages,
                tools=TOOLS,
                result_format="message"
            )
            choices = resp.output.get("choices", [])
            if not choices:
                answer = "AI 思考失败"
                break
            ai_msg = choices[0]["message"]

            # 不需要工具，直接输出最终答案
            if "tool_calls" not in ai_msg or not ai_msg["tool_calls"]:
                answer = ai_msg.get("content", "我是课堂分析助手")
                break

            # 执行工具调用
            tool = ai_msg["tool_calls"][0]
            func_name = tool["function"]["name"]
            args = json.loads(tool["function"]["arguments"])
            tool_result = ""

            if func_name == "tool_get_teacher_all_reports":
                tool_result = tool_get_teacher_all_reports(teacher_code)
            elif func_name == "tool_get_single_report_detail":
                tool_result = tool_get_single_report_detail(args.get("report_id"))
                update_teacher_long_memory(teacher_code, report_id=args.get("report_id"))
            elif func_name == "tool_get_class_students":
                tool_result = tool_get_class_students(args.get("class_id"))
                update_teacher_long_memory(teacher_code, class_id=args.get("class_id"))
            elif func_name == "tool_get_time_range_stats":
                tool_result = tool_get_time_range_stats(teacher_code, args.get("time_type", "7d"))
            elif func_name == "tool_get_class_keyframe":
                tool_result = tool_get_class_keyframe(args.get("report_id"))
            else:
                tool_result = "未知工具"

            # 把工具结果塞回上下文，继续反思
            messages.append(ai_msg)
            messages.append({
                "role": "tool",
                "content": tool_result,
                "name": func_name
            })

            current_round += 1

        # 轮次用完仍未得出结论
        if not answer:
            answer = "已为你查询多节课堂数据，你可以指定具体报告ID查看详细专注度分析。"

        # 保存会话
        history.append({"role": "assistant", "content": answer})
        title = question[:20] + "..." if len(question) > 20 else question
        save_session_messages(teacher_code, session_id, title, history)

        return answer

    except Exception as e:
        return f"错误：{str(e)}"

def get_teacher_long_memory(teacher_code: str):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT focus_class_ids, focus_report_ids, prefer_question_type
            FROM teacher_long_memory WHERE teacher_code = %s
        """, (teacher_code,))
        row = cursor.fetchone()
        cursor.close()
        db.close()
        if not row:
            return {"focus_class_ids":"","focus_report_ids":"","prefer_question_type":""}
        return row
    except:
        return {"focus_class_ids":"","focus_report_ids":"","prefer_question_type":""}
    
def update_teacher_long_memory(teacher_code: str, class_id=None, report_id=None, q_type=None):
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor(pymysql.cursors.DictCursor)

        # 先查原有
        old = get_teacher_long_memory(teacher_code)
        cls_list = old["focus_class_ids"].split(",") if old["focus_class_ids"] else []
        rep_list = old["focus_report_ids"].split(",") if old["focus_report_ids"] else []

        # 追加班级
        if class_id and str(class_id) not in cls_list:
            cls_list.append(str(class_id))
        # 追加报告
        if report_id and str(report_id) not in rep_list:
            rep_list.append(str(report_id))

        new_cls = ",".join(cls_list[:10])  # 最多存10个
        new_rep = ",".join(rep_list[:10])
        new_qtype = q_type if q_type else old["prefer_question_type"]

        # 存在则更新，不存在则插入
        cursor.execute("""
            INSERT INTO teacher_long_memory
            (teacher_code, focus_class_ids, focus_report_ids, prefer_question_type)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            focus_class_ids=%s, focus_report_ids=%s, prefer_question_type=%s, update_time=NOW()
        """, (teacher_code, new_cls, new_rep, new_qtype, new_cls, new_rep, new_qtype))

        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print("长期记忆更新失败", e)
        
def build_system_prompt_with_memory(teacher_code: str, base_prompt: str):
    mem = get_teacher_long_memory(teacher_code)
    extra = f"""
【用户长期习惯记忆】
常关注班级ID：{mem['focus_class_ids']}
常查看报告ID：{mem['focus_report_ids']}
偏好问题类型：{mem['prefer_question_type']}
后续回答优先结合该老师常用班级、常用报告，贴合使用习惯。
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