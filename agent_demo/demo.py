import dashscope
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from http import HTTPStatus
import json

# --- 1. 配置 Qwen API ---
dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"

# --- 2. 定义智能体之间的“共享笔记本” (State) ---
class AgentState(TypedDict):
    visual_info: str      # 视觉 Agent 的发现
    lesson_info: str      # 语境 Agent 的信息
    final_decision: str   # 大脑 Agent 的最终结论
    short_term_memory: List[str] # 新增：存放过去几分钟的结论

# --- 3. 定义各个 Agent 的逻辑 ---

def perception_node(state: AgentState):
    """
    视觉 Agent 节点。
    实际运行中，这里会调用你之前的 YOLO + Qwen-VL。
    """
    print("--- 视觉 Agent 正在分析图像... ---")
    # 模拟 VLM 的输出结果
    vlm_result = "观察到前排 3 名学生低头，手中握笔，眼神注视桌面。发现一名平时很努力的学生在睡觉。"
    return {"visual_info": vlm_result}

def context_node(state: AgentState):
    """
    语境 Agent 节点。
    实际运行中，这里会去检索你的课程 PDF 或数据库。
    """
    print("--- 语境 Agent 正在分析教学背景... ---")
    # 模拟知识库检索结果
    db_result = "当前课程时间：第 25 分钟。教学环节：随堂课堂练习（推导公式）。"
    return {"lesson_info": db_result}

# --- 1. 定义工具函数 (真实的 Python 函数) ---
def get_student_history(student_description: str):
    """
    模拟工具：根据学生描述查询其历史课堂表现。
    """
    # 实际项目中，这里可以查询数据库
    print(f"--- [工具调用] 正在查询历史数据: {student_description} ---")
    return "该生上周专注度排名全班前5%，今日可能是突发疲劳。"

# --- 2. 将工具信息封装，告诉大模型它有这个能力 ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_student_history",
            "description": "当发现某个学生行为异常时，查询其历史表现以辅助判断",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_description": {"type": "string", "description": "学生的特征描述，如'前排穿红衣服的男生'"}
                },
                "required": ["student_description"]
            }
        }
    }
]

def master_brain_node(state: AgentState):
    """
    大脑 Agent：利用 LLM 进行逻辑推导。
    """
    print("--- 调度大脑正在进行逻辑决策... ---")
    
    memory_context = "\n".join(state['short_term_memory'][-3:]) # 只看最近3条记录
    
    # 构建 Prompt
    messages = [
        {"role": "system", "content": f"你是一个助教 Agent。如果视觉信息模糊或需要更多证据，请调用工具查询历史。同时你也是一个带记忆的助教。历史记录：{memory_context}"},
        {"role": "user", "content": f"视觉信息: {state['visual_info']}\n背景: {state['lesson_info']}"}
    ]
    
    # 调用 Qwen-max 进行逻辑推理
    response = dashscope.Generation.call(
            model='qwen-max',
            messages=messages,
            tools=tools, # 传入工具定义
            result_format='message'
        )
    
    # 检查模型是否想调用工具
    if response.status_code == HTTPStatus.OK:
        message = response.output.choices[0].message
        if message.get("tool_calls"):
            # 工具调用逻辑
            tool_call = message["tool_calls"][0]
            args = json.loads(tool_call['function']['arguments'])
            tool_result = get_student_history(args.get('student_description'))
            
            # 二次推理
            follow_up_messages = messages + [
                message,
                {"role": "tool", "content": tool_result, "name": tool_call['function']['name']}
            ]
            final_res = dashscope.Generation.call(model='qwen-max', messages=follow_up_messages, result_format='message')
            res_text = final_res.output.choices[0].message['content']
        else:
            res_text = message['content']
    else:
        res_text = "决策引擎调用失败。"
        
    # 【关键更新】：将本次结论存入记忆，传给下一轮
    new_memory = state.get('short_term_memory', []) + [f"最新记录: {res_text[:50]}..."]
    return {"final_decision": res_text, "short_term_memory": new_memory}

# --- 4. 使用 LangGraph 构建工作流图 ---

# 初始化图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("vision_agent", perception_node)
workflow.add_node("context_agent", context_node)
workflow.add_node("master_brain", master_brain_node)

# 连线逻辑：感知 -> 背景 -> 大脑 -> 结束
workflow.set_entry_point("vision_agent")
workflow.add_edge("vision_agent", "context_agent")
workflow.add_edge("context_agent", "master_brain")
workflow.add_edge("master_brain", END)

# 编译成可执行应用
app = workflow.compile()

# --- 5. 运行 Demo ---
if __name__ == "__main__":
    print("=== 启动智慧课堂多智能体系统 ===\n")
    initial_state = {
        "visual_info": "",
        "lesson_info": "",
        "final_decision": "",
        "short_term_memory": ["10分钟前：全班状态良好，班长正在认真记笔记。"]
    }
    
    # 运行图逻辑
    result = app.invoke(initial_state)
    
    print("\n" + "="*30)
    print("【最终决策报告】")
    print(result['final_decision'])
    print("\n【当前记忆池状态】")
    print(result['short_term_memory'])
    print("="*30)