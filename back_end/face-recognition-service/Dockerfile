# 使用轻量级 Python 镜像
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 升级 pip
RUN python3 -m pip install --upgrade pip

# 先复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有源代码
COPY . .

# 暴露端口
EXPOSE 5003

# 启动命令
CMD ["python3", "app.py"]