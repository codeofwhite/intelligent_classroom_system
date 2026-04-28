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

# 🔥 修复：本地 MinIO 地址
MINIO_CLIENT = Minio(
    "localhost:9000",  # 改为本地！
    access_key="admin",
    secret_key="password123",
    secure=False
)
BUCKET_NAME = "video-bucket"

CHAT_HISTORY = []

# ========================
# 工具 1：获取老师历史报告（真实MySQL）
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
               for i, row in enumerate(rows)]
        return f"您的历史课堂（共{len(rows)}节）：\n" + "\n".join(res)
    except Exception as e:
        return f"获取历史失败：{str(e)}"

# ========================
# 工具 2：获取单节课详情（修复版）
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

        # 尝试获取文件
        try:
            data = MINIO_CLIENT.get_object(BUCKET_NAME, report['minio_json_path'])
            stats = json.loads(data.data)
            counts = stats["behavior_counts"]
        except:
            return (f"报告ID {report_id} 课堂概况（无法读取详细文件）：\n"
                    "✅ 已完成课堂分析\n"
                    "可查看专注度、分心行为、举手次数等数据")

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
# 工具定义
# ========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_get_teacher_all_reports",
            "description": "获取老师所有历史课堂记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "teacher_code": {"type": "string"}
                },
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
                "properties": {
                    "report_id": {"type": "integer"}
                },
                "required": ["report_id"]
            }
        }
    }
]

# ========================
# 核心聊天 Agent
# ========================
def chat_agent(question: str, teacher_code: str = "T2025001"):
    try:
        CHAT_HISTORY.append({"role": "user", "content": question})

        messages = [
            {"role": "system", "content": "你是课堂分析助手，必须调用工具获取真实数据回答老师问题。"}
        ] + CHAT_HISTORY[-5:]

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

            print(f"\n🤖 LLM 自主调用工具：{func_name}")
            print(f"📥 参数：{args}")

            if func_name == "tool_get_teacher_all_reports":
                tool_result = tool_get_teacher_all_reports(teacher_code)
            elif func_name == "tool_get_single_report_detail":
                tool_result = tool_get_single_report_detail(args["report_id"])
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

        CHAT_HISTORY.append({"role": "assistant", "content": answer})
        return answer

    except Exception as e:
        return f"错误：{str(e)}"

# ========================
# 测试入口
# ========================
if __name__ == "__main__":
    print("==================================================")
    print("✅ 课堂AI聊天助手（稳定完美版）")
    print("💡 输入 exit 退出")
    print("==================================================")
    while True:
        q = input("\n你：")
        if q.lower() == "exit":
            break
        ans = chat_agent(q, teacher_code="T2025001")
        print("\nAI：", ans)