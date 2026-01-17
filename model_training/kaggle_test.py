import os
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. 初始化并认证
api = KaggleApi()
api.authenticate()

# 2. 搜索你毕设相关的关键字（比如 "Student Behavior"）
print("正在搜索课堂行为相关数据集...")
datasets = api.dataset_list(search='classroom behavior')

for ds in datasets:
    print(f"数据集 ID: {ds.ref} | 标题: {ds.title}")

# 3. 尝试下载一个很小的文件测试 (可选)
# api.dataset_download_file('whiffe/scb-05-dataset', 'data.yaml', path='./')