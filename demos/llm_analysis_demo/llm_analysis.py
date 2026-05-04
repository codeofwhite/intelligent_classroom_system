import os
import dashscope
import pandas as pd

def summarize_behavior(file_path):
    df = pd.read_csv(file_path)
    
    # 1. 过滤掉置信度太低的数据（比如低于0.5的）
    df = df[df['confidence'] > 0.5]
    
    # 2. 统计每个学生的行为频次
    summary = df.groupby(['student_id', 'behavior_label']).size().unstack(fill_value=0)
    
    # 3. 转化为文字描述
    description = ""
    for student_id, row in summary.iterrows():
        behavior_str = ", ".join([f"{label}{count}次" for label, count in row.items()])
        description += f"学生{student_id}的行为表现为：{behavior_str}。\n"
    
    return description

# 获取统计后的摘要
csv_summary = summarize_behavior('classroom_analysis_results.csv')
print(csv_summary)

def generate_report(summary_text):
    dashscope.api_key = 'sk-06abd7a7eb514b3ebd611412f0dc3531'
    
    prompt = f"""
    以下是从智能教室系统提取的 CSV 统计数据：
    {summary_text}
    
    请根据这些数据生成一份专业的课堂行为分析报告。报告应包含：
    1. 课堂整体纪律评估。
    2. 针对表现异常的学生（如回头次数过多）给出预警。
    3. 给老师的教学改进建议。
    """
    
    response = dashscope.Generation.call(
        model="qwen-turbo",  # 使用最便宜的模型
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    if response.status_code == 200:
        return response.output.text
    else:
        return f"生成失败: {response.message}"

# 生成最终报告
report = generate_report(csv_summary)
print(report)