import dashscope

dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"

def analyze_class_report(behavior_data, class_info):
    # ========== 这是标准 Agent，不是普通问答 ==========
    system_prompt = """
你是【课堂行为分析智能Agent】，具备专业的课堂分析能力。
你必须严格按照4个步骤完成分析：
1. 量化课堂行为数据
2. 诊断学生状态与问题
3. 分析原因
4. 给出可落地的教学建议

输出使用Markdown，分模块展示：
- 课堂概况
- 数据分析
- 问题诊断
- 教学建议
"""

    # 关键：把字典变成干净文本，不报错！
    prompt = f"""
【课堂信息】
班级：{class_info['class_name']}
节次：{class_info['lesson_section']}

【学生行为统计】
举手：{behavior_data['举手']}
看书：{behavior_data['看书']}
写字：{behavior_data['写字']}
使用手机：{behavior_data['使用手机']}
低头做其他事情：{behavior_data['低头做其他事情']}
睡觉：{behavior_data['睡觉']}

请以课堂分析Agent身份，完成完整课堂分析报告。
"""

    response = dashscope.Generation.call(
        model="qwen-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        result_format="message"
    )

    try:
        return response.output.choices[0].message.content.strip()
    except:
        return "AI 分析暂时不可用，请稍后再试。"