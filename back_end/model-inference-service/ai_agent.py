import dashscope
import json
import os
import time
import requests
from typing import Dict, Any
from minio import Minio

# 短期：本次课堂会话记忆
session_memory = []
# 长期：班级课程记忆 key:class_id+course_name
course_long_memory: Dict[str, list] = {}

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

MINIO_CLIENT = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    secure=False
)
BUCKET_NAME = "video-bucket"

# 记忆写入
def add_memory(content: str):
    session_memory.append(content)
    if len(session_memory) > 10:
        session_memory.pop(0)

# 获取拼接记忆
def get_memory_context() -> str:
    if not session_memory:
        return "暂无本次课堂分析上下文记忆"
    return "\n".join([f"【历史记录】{item}" for item in session_memory])

# ========================
# 工具定义（已改成真实接口！）
# ========================
def tool_get_yolo_data(behavior_data):
    print("[工具] 正在获取 YOLO 行为统计...")

    # ✅ 关键：告诉 AI 这是【次数】不是【人数】
    res = """
【注意：以下为AI视觉检测次数，非学生人数】
学生行为检测次数统计：
"""
    for k, v in behavior_data.items():
        res += f"- {k}：{v} 次\n"

    print("[工具] YOLO 数据获取完成 ✅")
    return res

def tool_vlm_image_analysis(image_minio_path: str):
    if not image_minio_path:
        print("[VLM] 未找到关键帧图片，跳过")
        return "未检测到有效课堂关键帧图像"

    print(f"[VLM] 正在分析 MinIO 图片：{image_minio_path}")
    try:
        import os
        TMP_DIR = "./tmp_vlm"
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        tmp_file = os.path.join(TMP_DIR, "tmp_keyframe.jpg")
        response = MINIO_CLIENT.get_object(BUCKET_NAME, image_minio_path)
        with open(tmp_file, "wb") as f:
            f.write(response.read())

        requests.adapters.DEFAULT_TIMEOUT = 15
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "详细描述这张课堂图片：学生状态、坐姿、专注度、课堂情况"},
                    {"image": f"file://{tmp_file}"}
                ]
            }
        ]

        resp = dashscope.MultiModalConversation.call(
            model="qwen-vl-max",
            messages=messages
        )

        if not resp or not resp.output.choices:
            return "VLM 视觉分析无结果"

        content = resp.output.choices[0].message.content[0]["text"]
        return f"🎥 课堂画面分析：{content}"

    except Exception as e:
        print(f"[VLM] 错误：{e}")
        return f"VLM 分析失败：{str(e)}"

# ========================
# ✅ 【重大修改】这里改成调用你真实的课表接口！
# ========================
def tool_get_real_course_schedule(teacher_code: str):
    try:
        print(f"[工具] 正在从后端获取老师【{teacher_code}】真实课表...")
        
        # 调用你自己的后端接口
        res = requests.post(
            "http://localhost:5002/api/teacher/course_schedule",
            json={"teacher_code": teacher_code},
            timeout=10
        )
        data = res.json()
        schedule_list = data.get("list", [])

        if not schedule_list:
            return "📅 该老师暂无课程安排"

        week_map = {1:"周一",2:"周二",3:"周三",4:"周四",5:"周五",6:"周六",7:"周日"}
        lines = ["📅 老师真实课程安排："]
        for item in schedule_list:
            wd = week_map.get(item["week_day"], "未知")
            sec = item["section"]
            cls = item["class_name"]
            cou = item["course_name"]
            room = item["classroom"]
            lines.append(f"• {wd} 第{sec}节｜{cls} - {cou}｜{room}")

        result = "\n".join(lines)
        print("[工具] 真实课表获取完成 ✅")
        return result

    except Exception as e:
        print(f"[工具] 获取课表失败：{e}")
        return "获取真实课表失败，使用基础信息"

def tool_dispatcher(behavior_data, frame_path, teacher_code):
    print("\n========================================")
    print("🤖 AGENT 开始调度所有工具...")
    tool_res = []
    tool_res.append(tool_get_yolo_data(behavior_data))
    tool_res.append(tool_vlm_image_analysis(frame_path))
    tool_res.append(tool_get_real_course_schedule(teacher_code))  # ✅ 真实接口
    print("🤖 AGENT 所有工具调用完成 ✅")
    print("========================================\n")
    return "\n\n".join(tool_res)

# ========================
# 双 Agent
# ========================
def perception_agent(tool_result: str, memory_ctx: str):
    print("\n🧠 启动 PERCEPTION AGENT（感知层）...")
    start = time.time()

    system = """
你是Perception Agent课堂感知智能体，只做客观事实提取，不主观评价。
输入：YOLO行为数据、VLM画面描述、真实课程信息、历史记忆
输出：客观课堂状态、学生行为分布、画面直观现象
"""
    user = f"""
【历史记忆上下文】
{memory_ctx}

【工具返回全部数据】
{tool_result}

请输出客观课堂感知摘要。
"""
    resp = dashscope.Generation.call(
        model="qwen-max",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        result_format="message"
    )
    res = resp.output.choices[0].message.content.strip()
    add_memory(f"感知Agent输出：{res}")

    print(f"🧠 PERCEPTION AGENT 完成，耗时：{round(time.time()-start,1)}s ✅")
    return res

def conclusion_agent(perception_content: str, class_info: dict, course_name: str):
    print("\n📝 启动 CONCLUSION AGENT（决策层）...")
    start = time.time()

    system = """
你是Conclusion Agent课堂决策大脑，结合感知数据+真实课程表做深度分析。
必须完成：
1. 课堂整体状态综合评估
2. 分心行为占比与问题诊断
3. 结合真实课程、班级、学科特点分析
4. 判断课堂节奏与教学适配性
5. 输出专业Markdown报告 + 教学改进建议
"""
    memory_ctx = get_memory_context()
    user = f"""
【班级基础信息】
班级：{class_info['class_name']}
上课节次：{class_info['lesson_section']}
授课课程：{course_name}

【历史记忆】
{memory_ctx}

【感知层客观结论】
{perception_content}

请输出完整标准化课堂Agent分析报告。
"""
    resp = dashscope.Generation.call(
        model="qwen-max",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        result_format="message"
    )
    final_report = resp.output.choices[0].message.content.strip()
    add_memory(f"最终报告生成完成")

    print(f"📝 CONCLUSION AGENT 完成，耗时：{round(time.time()-start,1)}s ✅")
    return final_report

# ========================
# 总入口（已升级）
# ========================
def analyze_class_report(
    behavior_data,
    class_info: dict,
    teacher_code: str,
    course_name: str = "常规文化课",
    frame_path: str = None
):
    print("\n" + "="*60)
    print("🚀 课堂行为分析 AGENT 系统启动（真实数据版）")
    print("="*60)

    total_start = time.time()

    print("\n[1/5] 清空会话记忆...")
    session_memory.clear()

    # ✅ 调度工具（含真实课表）
    print("[2/5] 调度所有工具（YOLO + VLM + 真实课表）...")
    tool_all_data = tool_dispatcher(behavior_data, frame_path, teacher_code)

    print("[3/5] 加载历史记忆上下文...")
    memory_ctx = get_memory_context()

    print("[4/5] 启动感知智能体...")
    perception_res = perception_agent(tool_all_data, memory_ctx)

    print("[5/5] 启动决策智能体生成报告...")
    final_report = conclusion_agent(perception_res, class_info, course_name)

    print("\n" + "="*60)
    print(f"✅ AGENT 系统全部完成！总耗时：{round(time.time() - total_start, 1)} 秒")
    print("="*60 + "\n")

    return final_report