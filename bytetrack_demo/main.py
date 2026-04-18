import cv2
import csv
import time
import sys
import numpy as np
from collections import deque
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# 1. 动态添加 ByteTrack 路径
sys.path.append('./ByteTrack') 
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

class ByteTrackArgs:
    track_thresh = 0.2
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

def get_color(category_id):
    """为 14 种课堂行为生成高区分度颜色 (BGR 格式)"""
    # 按照 14 种行为预设的调色盘
    colors = [
        (255, 0, 0),      # 0: 纯蓝
        (0, 255, 0),      # 1: 纯绿
        (0, 0, 255),      # 2: 纯红
        (255, 255, 0),    # 3: 青色
        (255, 0, 255),    # 4: 品红
        (0, 255, 255),    # 5: 纯黄
        (128, 0, 0),      # 6: 深蓝
        (0, 128, 0),      # 7: 深绿
        (0, 0, 128),      # 8: 深红
        (128, 128, 0),    # 9: 鸭羽绿
        (128, 0, 128),    # 10: 紫色
        (0, 128, 128),    # 11: 橄榄色
        (255, 128, 0),    # 12: 橙色
        (0, 128, 255),    # 13: 天蓝色
    ]
    # 使用取余算法，即使类别超过 14 也不会报错
    return colors[category_id % len(colors)]

def analyze_classroom_data(csv_path):
    # 设置中文字体（如果是中文标签需要，没有可忽略）
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据
    df = pd.read_csv(csv_path)
    if df.empty:
        print("数据为空，无法分析")
        return

    # --- 1. 计算每个学生的“专注度评分” ---
    # 定义权重：听讲/书写=+1，举手=+2，玩手机/睡觉=-5，其他=0
    weight_map = {
        'Listening': 1, 'Writing': 1, 'Raising Hand': 2, 
        'Phone': -5, 'Sleeping': -5, 'Leaning': -1
    }
    # 这里的标签名要对应你 SCB-Dataset 里的实际 names 列表

    df['score'] = df['behavior_label'].map(weight_map).fillna(0)
    student_scores = df.groupby('student_id')['score'].sum().sort_values(ascending=False)
    
    # 归一化到 0-100 分
    student_scores = (student_scores - student_scores.min()) / (student_scores.max() - student_scores.min()) * 100

    # 绘制排行榜
    plt.figure(figsize=(10, 6))
    student_scores.head(10).plot(kind='barh', color='skyblue')
    plt.title('课堂专注度 Top 10 学生排行榜')
    plt.xlabel('综合评分')
    plt.savefig('student_rank.png')

    # --- 2. 异常行为热力图 (Heatmap) ---
    if 'cx' in df.columns and 'cy' in df.columns:
        # 过滤出负面行为
        bad_df = df[df['score'] < 0]
        
        plt.figure(figsize=(10, 6))
        # 简单使用 hexbin 或 hist2d 模拟热力分布
        plt.hexbin(bad_df['cx'], bad_df['cy'], gridsize=20, cmap='YlOrRd')
        plt.colorbar(label='异常行为频次')
        plt.title('教室内异常行为空间分布热力图')
        # 翻转 Y 轴，因为图像坐标原点在左上角
        plt.gca().invert_yaxis()
        plt.savefig('classroom_heatmap.png')

    print("✅ 进阶分析图表已生成：排行榜与热力图。")

    # 2. 基础信息统计
    total_frames = df['frame_id'].max()
    unique_students = df['student_id'].nunique()
    
    print("-" * 30)
    print(f"📊 课堂综合分析报告")
    print(f"总处理帧数: {total_frames}")
    print(f"检测到总人数: {unique_students}")
    print("-" * 30)

    # 3. 行为占比统计 (饼图)
    behavior_counts = df['behavior_label'].value_counts()
    
    plt.figure(figsize=(10, 6))
    behavior_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('课堂行为总体分布占比')
    plt.ylabel('') # 隐藏 y 轴标签
    plt.savefig('behavior_distribution.png')
    print("✅ 行为分布饼图已保存为: behavior_distribution.png")

    # 4. 行为随时间变化趋势 (柱状图)
    # 将视频分为 10 个阶段进行观察
    df['stage'] = pd.cut(df['frame_id'], bins=10, labels=[f"阶段{i+1}" for i in range(10)])
    stage_analysis = df.groupby(['stage', 'behavior_label']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    stage_analysis.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('课堂行为随时间变化趋势')
    plt.xlabel('视频阶段')
    plt.ylabel('出现频次')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('behavior_trend.png')
    print("✅ 行为趋势图已保存为: behavior_trend.png")

    plt.show()

def main(video_path, model_path, output_path):
    model = YOLO(model_path, task='detect')
    tracker = BYTETracker(ByteTrackArgs())
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 25
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    track_history = {}
    print("开始推理")

    behavior_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. YOLO 检测 (使用 OpenVINO)
        results = model.predict(frame, device="cpu", conf=0.3, verbose=False)
        det = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        if len(det) > 0:
            # 2. ByteTrack 跟踪
            online_targets = tracker.update(det[:, :5], [height, width], [height, width])

            # 【新增】实时人数显示
            current_count = len(online_targets)
            cv2.putText(frame, f"Students: {current_count}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            for t in online_targets:
                tlbr = t.tlbr
                tid = t.track_id
                
                # 计算中心点 cx, cy
                cx = (tlbr[0] + tlbr[2]) / 2
                cy = (tlbr[1] + tlbr[3]) / 2
                
                # 类别匹配逻辑
                tx, ty = cx, cy # 使用已有的中心点
                
                # 在 det 中找距离最近的检测框作为其类别来源
                best_cls = 0
                min_dist = float('inf')
                
                for d in det:
                    dx = (d[0] + d[2]) / 2
                    dy = (d[1] + d[3]) / 2
                    dist = (tx - dx)**2 + (ty - dy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_cls = int(d[5])

                # 标签与平滑
                label_name = results[0].names[best_cls]
                if tid not in track_history:
                    track_history[tid] = deque(maxlen=15)
                track_history[tid].append(label_name)
                smooth_label = max(set(track_history[tid]), key=track_history[tid].count)

                behavior_data.append([
                    frame_count, 
                    tid, 
                    smooth_label, 
                    round(float(t.score), 2), 
                    int(cx), # 存入整数坐标即可
                    int(cy),
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])

                # --- 3. 绘制 (根据类别选择颜色) ---
                color = get_color(best_cls)
                x1, y1, x2, y2 = map(int, tlbr)
                
                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 画标签背景
                text = f"ID:{tid} {smooth_label}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        out.write(frame)
        
    # --- 【新增】推理结束后，保存为 CSV 文件 ---
    csv_filename = "classroom_analysis_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['frame_id', 'student_id', 'behavior_label', 'confidence', 'cx', 'cy', 'timestamp'])
        # 写入数据
        writer.writerows(behavior_data)
        
    print(f"✅ 行为轨迹数据已保存至: {csv_filename}")
        
    cap.release()
    out.release()
    print("\n✅ 处理完成！")

if __name__ == "__main__":
    video_input = "demo/37233165569-1-192 - Trim.mp4"
    csv_output = "classroom_analysis_results.csv"
    main(video_input, "best_last_openvino_model", "output_xpu_colored.mp4")
    analyze_classroom_data(csv_output)