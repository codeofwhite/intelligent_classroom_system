import dashscope

dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"

def analyze_class_report(behavior_data, class_info):
    prompt = f"""
你是智慧课堂助教，请根据以下学生行为数据，生成专业、简洁、有条理的课堂分析报告。

【课堂信息】
班级：{class_info['class_name']}
节次：{class_info['lesson_section']}

【学生行为统计】
举手：{behavior_data['举手']}
看书：{behavior_data['看书']}
写字：{behavior_data['写字']}
使用手机：{behavior_data['使用手机']}
低头：{behavior_data['低头做其他事情']}
睡觉：{behavior_data['睡觉']}

请输出：
1. 课堂整体状态
2. 主要问题
3. 教学建议
"""

    response = dashscope.Generation.call(
        model="qwen-max",
        messages=[{"role": "user", "content": prompt}],
        result_format="message"
    )

    try:
        return response.output.choices[0].message.content
    except:
        return "AI 分析暂时不可用，请稍后再试。"