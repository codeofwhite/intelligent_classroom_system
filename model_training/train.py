# train.py
import os

# 检查 GPU 是否可用 (Kaggle 环境下)
import torch
print(f"GPU 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

# 模拟一个训练循环或直接写你的 YOLO 逻辑
print("开始毕设模型训练任务...")
# model.train(...)
print("训练完成！")