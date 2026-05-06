# ==============================================
# 本地运行！一次性下载 YOLOv8 / v10 / v11 全部模型
# ==============================================
import torch
import os

# 下载所有你需要的模型
model_urls = {
    # YOLOv8
    # "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    # "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    # "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",

    # YOLOv10
    "yolov10n.pt": "https://github.com/jameslahm/yolov10/releases/download/v1.1/yolov10n.pt",
    "yolov10s.pt": "https://github.com/jameslahm/yolov10/releases/download/v1.1/yolov10s.pt",
    "yolov10m.pt": "https://github.com/jameslahm/yolov10/releases/download/v1.1/yolov10m.pt",

    # YOLOv11 (官方最新发布)
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
}

for name, url in model_urls.items():
    print(f"正在下载: {name}...")
    destination = os.path.join("./", name)
    
    try:
        # 【关键修改】：只下载文件，不进行 torch.load() 加载
        torch.hub.download_url_to_file(url, destination)
        print(f"✅ {name} 已成功保存至本地。")
    except Exception as e:
        print(f"❌ {name} 下载出错: {e}")

print("✅ 所有模型下载完成！全部保存在当前文件夹中")