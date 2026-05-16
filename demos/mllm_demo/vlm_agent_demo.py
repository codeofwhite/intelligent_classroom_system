import os
import dashscope
from dashscope import MultiModalConversation
from collections import Counter
import pandas as pd
import io

# 配置 API Key（从环境变量读取）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

def analyze_first_frame(image_path, csv_data):
    # 1. 简单处理 CSV 数据获取统计信息
    df = pd.read_csv(io.StringIO(csv_data))
    stats = df['behavior_label'].value_counts().to_dict()
    total = len(df)
    
    # 提取异常点（手机和睡觉）
    anomalies = df[df['behavior_label'].isin(['Useing-Phone', 'Sleeping'])]
    anomaly_count = len(anomalies)

    # 2. 构造给 MLLM 的文本描述
    yolo_summary = f"""
【实时视觉统计 - Frame 1】
- 检测到学生总数: {total}
- 学习状态: Writing({stats.get('Writing', 0)}), Reading({stats.get('Reading', 0)})
- 异常风险: Useing-Phone({stats.get('Useing-Phone', 0)}), Sleeping({stats.get('Sleeping', 0)}), Head-down({stats.get('Head-down', 0)})
- 整体抬头率预测: {((stats.get('Writing', 0) + stats.get('Reading', 0)) / total * 100):.1f}%
"""

    prompt = f"""
你是一名课堂行为分析专家。请根据提供的图片和与之对应的 YOLO 检测统计数据进行分析。

{yolo_summary}

【任务】:
1. **视觉一致性校验**: 图片中的课堂氛围与统计数据（{anomaly_count}个异常行为）是否吻合？
2. **细节捕捉**: 请观察图片中被标记为 'Useing-Phone' 的区域，他们是真的在违规使用手机，还是在利用手机查阅资料或扫描二维码？
3. **综合评价**: 仅基于这一帧的画面和数据，评价当前的课堂秩序。
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": prompt}
            ]
        }
    ]

    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    return response

# --- 测试输入 ---
csv_raw_data = """
frame_id,student_id,behavior_label,confidence,cx,cy,timestamp
1,1,Writing,0.95,987,191,2026-04-14 17:35:41
1,2,Reading,0.95,106,466,2026-04-14 17:35:41
1,3,Useing-Phone,0.95,334,422,2026-04-14 17:35:41
1,4,Writing,0.95,658,283,2026-04-14 17:35:41
1,5,Useing-Phone,0.94,241,461,2026-04-14 17:35:41
1,6,Writing,0.94,176,321,2026-04-14 17:35:41
1,7,Useing-Phone,0.93,503,314,2026-04-14 17:35:41
1,8,Useing-Phone,0.93,623,218,2026-04-14 17:35:41
1,9,Useing-Phone,0.93,435,183,2026-04-14 17:35:41
1,10,Reading,0.92,93,347,2026-04-14 17:35:41
1,11,Useing-Phone,0.92,361,218,2026-04-14 17:35:41
1,12,Useing-Phone,0.92,589,141,2026-04-14 17:35:41
1,13,Reading,0.91,489,105,2026-04-14 17:35:41
1,14,Useing-Phone,0.91,679,116,2026-04-14 17:35:41
1,15,Head-down,0.91,655,160,2026-04-14 17:35:41
1,16,Sleeping,0.91,564,249,2026-04-14 17:35:41
1,17,Reading,0.91,131,277,2026-04-14 17:35:41
1,18,Writing,0.9,497,160,2026-04-14 17:35:41
1,19,Reading,0.9,798,90,2026-04-14 17:35:41
1,20,Reading,0.9,743,72,2026-04-14 17:35:41
1,21,Useing-Phone,0.9,626,123,2026-04-14 17:35:41
1,22,Useing-Phone,0.9,499,202,2026-04-14 17:35:41
1,23,Reading,0.9,438,117,2026-04-14 17:35:41
1,24,Reading,0.89,724,183,2026-04-14 17:35:41
1,25,Reading,0.89,686,195,2026-04-14 17:35:41
1,26,Useing-Phone,0.89,836,84,2026-04-14 17:35:41
1,27,Head-down,0.88,422,246,2026-04-14 17:35:41
1,28,Reading,0.87,861,109,2026-04-14 17:35:41
1,29,Sleeping,0.86,778,69,2026-04-14 17:35:41
1,30,Reading,0.86,895,97,2026-04-14 17:35:41
1,31,Head-down,0.85,512,81,2026-04-14 17:35:41
1,32,Reading,0.44,399,125,2026-04-14 17:35:41
"""

# 替换成你本地的第一帧截图
test_image_path = "frames/frame_1.png" 

print("正在通过 MLLM 审计第一帧数据...")
res = analyze_first_frame(test_image_path, csv_raw_data)

if res.status_code == 200:
    print("\n📝 MLLM 行为审计报告：")
    print(res.output.choices[0].message.content[0]['text'])
else:
    print(f"Error: {res.message}")