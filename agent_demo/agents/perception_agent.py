import cv2
import dashscope
from ultralytics import YOLO
from http import HTTPStatus

# --- 1. 配置区域 ---
dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"
MODEL_PATH = 'yolov8n.pt'  # 首次运行会自动下载
VIDEO_PATH = 'demo/test_university_short.mp4'  # 替换成你的视频路径

yolo_model = YOLO(MODEL_PATH)

def analyze_classroom_with_qwen(image_path, student_count):
    """调用 Qwen-VL 进行多模态分析"""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": f"当前是智慧课堂监控。YOLO检测到有{student_count}人。请结合图像分析：学生是在认真听课、睡觉还是玩手机？如果发现违纪或疲劳，请给出助教建议。"}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content[0]['text']
    else:
        return f"Error: {response.message}"

# --- 3. 主流程 ---
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    
    print("开始处理视频...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 每 30 帧处理一次 (假设视频是30fps，即每秒分析一次)
        if frame_idx % 30 == 0:
            # A. YOLO 检测
            results = yolo_model(frame, verbose=False)
            # 过滤出 person 类别 (ID 为 0)
            persons = [box for box in results[0].boxes if int(box.cls) == 0]
            count = len(persons)
            
            # B. 保存当前帧为临时图片，供 VLM 读取
            tmp_img = f"frame_{frame_idx}.jpg"
            cv2.imwrite(tmp_img, frame)
            
            # C. 调用 Agent 决策
            print(f"\n[时间点: {frame_idx//30}秒] 检测到人数: {count}")
            decision = analyze_classroom_with_qwen(tmp_img, count)
            print(f"Agent 建议: {decision}")
            
        frame_idx += 1

    cap.release()
    print("处理完成。")

if __name__ == "__main__":
    main()

