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

dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"

MINIO_CLIENT = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)
BUCKET_NAME = "video-bucket"

# 记忆写入
def add_memory(content: str):
    session_memory.append(content)
    # 限制记忆长度，防止超长
    if len(session_memory) > 10:
        session_memory.pop(0)

# 获取拼接记忆
def get_memory_context() -> str:
    if not session_memory:
        return "暂无本次课堂分析上下文记忆"
    return "\n".join([f"【历史记录】{item}" for item in session_memory])

# ========================
# 工具定义
# ========================
def tool_get_yolo_data(behavior_data):
    print("[工具] 正在获取 YOLO 行为统计...")
    res = f"YOLO行为检测统计：\n{json.dumps(behavior_data, ensure_ascii=False, indent=2)}"
    print("[工具] YOLO 数据获取完成 ✅")
    return res

def tool_vlm_image_analysis(image_minio_path: str):
    if not image_minio_path:
        print("[VLM] 未找到关键帧图片，跳过")
        return "未检测到有效课堂关键帧图像"

    print(f"[VLM] 正在分析 MinIO 图片：{image_minio_path}")
    try:
        # 1. 创建临时目录
        import os
        TMP_DIR = "./tmp_vlm"
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        # 2. 生成临时文件名
        tmp_file = os.path.join(TMP_DIR, "tmp_keyframe.jpg")

        # 3. 从 MinIO 下载到本地临时文件
        response = MINIO_CLIENT.get_object(BUCKET_NAME, image_minio_path)
        with open(tmp_file, "wb") as f:
            f.write(response.read())

        # 4. 调用 Qwen-VL（正确格式！）
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

def tool_get_course_info(course_name: str, class_name: str):
    print("[工具] 获取课程信息...")
    res = f"课程背景信息：班级{class_name}，当前授课课程：{course_name}，需结合学科特点评估课堂状态"
    print("[工具] 课程信息获取完成 ✅")
    return res

def tool_dispatcher(behavior_data, frame_path, course_name, class_name):
    print("\n========================================")
    print("🤖 AGENT 开始调度所有工具...")
    tool_res = []
    tool_res.append(tool_get_yolo_data(behavior_data))
    tool_res.append(tool_vlm_image_analysis(frame_path))
    tool_res.append(tool_get_course_info(course_name, class_name))
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
输入：YOLO行为数据、VLM画面描述、课程信息、历史记忆
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
你是Conclusion Agent课堂决策大脑，结合感知数据+课程场景做深度分析。
必须完成：
1. 课堂整体状态综合评估
2. 分心行为占比与问题诊断
3. 结合当前课程学科内容，分析课堂节奏、知识点讲解适配性
4. 判断课堂互动是否不足、内容是否重复冗余
5. 输出结构化Markdown专业报告+可落地教学改进建议
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
# 总入口
# ========================
def analyze_class_report(
    behavior_data,
    class_info: dict,
    course_name: str = "常规文化课",
    frame_path: str = None
):
    print("\n" + "="*60)
    print("🚀 课堂行为分析 AGENT 系统启动")
    print("="*60)

    total_start = time.time()

    # 清空记忆
    print("\n[1/5] 清空会话记忆...")
    session_memory.clear()

    # 调用工具
    print("[2/5] 调度所有工具（YOLO + VLM + 课程信息）...")
    tool_all_data = tool_dispatcher(behavior_data, frame_path, course_name, class_info["class_name"])

    # 记忆上下文
    print("[3/5] 加载历史记忆上下文...")
    memory_ctx = get_memory_context()

    # 感知Agent
    print("[4/5] 启动感知智能体...")
    perception_res = perception_agent(tool_all_data, memory_ctx)

    # 决策Agent
    print("[5/5] 启动决策智能体生成报告...")
    final_report = conclusion_agent(perception_res, class_info, course_name)

    # 结束
    print("\n" + "="*60)
    print(f"✅ AGENT 系统全部完成！总耗时：{round(time.time() - total_start, 1)} 秒")
    print("="*60 + "\n")

    return final_report