import cv2
import json
import os

class RagAnalysisEngine:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.points = self.data['knowledge_points']

    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return duration

    def get_current_context(self, current_sec, total_duration):
        """
        核心逻辑：支持绝对秒数匹配 或 进度百分比匹配
        """
        # 计算当前进度百分比
        progress = (current_sec / total_duration) * 100
        
        for point in self.points:
            start, end = point['time_range']
            # 如果老师输入的是 [0, 1]，我们视为百分比；如果是 [0, 600]，视为秒数
            if end <= 1.0: # 百分比模式
                if (start * 100) <= progress <= (end * 100):
                    return point
            else: # 绝对秒数模式
                if start <= current_sec <= end:
                    return point
        return None

# --- 运行示例 ---
video_path = "test.mp4"
json_path = "assets/test_math.json"

engine = RagAnalysisEngine(json_path)
total_t = engine.get_video_duration(video_path)

# 模拟：视频运行到第 100 秒
current_time = 800 
context = engine.get_current_context(current_time, total_t)

if context:
    # 结合你的 YOLO 日志结果（模拟数据）
    yolo_log = "检测到抬头率45%，3人低头"
    
    prompt = f"""
    【课程背景】：{engine.data['course_info']['name']}
    【当前环节】：{context['topic']} (难度：{context['difficulty']})
    【教学目标】：{context['content']}
    【实时监测】：{yolo_log}
    【任务】：结合背景分析学生状态并给老师建议。
    """
    print(prompt)