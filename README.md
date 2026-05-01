# Intelligent Classroom System (基于计算机视觉的智能课堂系统)

本项目是一款采用多端分离架构构建的智能化教学管理平台。系统通过集成前沿的计算机视觉算法，实现课堂状态的实时感知、行为分析及自动化考勤，旨在为教师提供客观的教学反馈，同时为家长提供透明的课堂参与渠道。

## 项目架构

系统主要由以下三个部分组成：

* **`back_end`**: 提供核心业务逻辑、数据库交互及视频处理 API。
* **`front_end` (Client)**: 面向学生与家长。
    - Admin 终端：面向教师与管理者，涵盖用户画像管理、课程智能排期及多维教学数据看板。
    - Client 终端：面向学生与家长，提供个性化课堂表现反馈及实时状态查询。
    - Terminal 终端：部署于教室现场的硬件采集端，负责实时推流、人脸识别签到及原始行为特征提取。

---

## 项目目录结构

```
intelligent_classroom_system/
├── README.md                          # 项目说明文档
├── .gitignore                         # Git 忽略配置
│
├── back_end/                          # 后端服务
│   ├── docker-compose.yml             # Docker 编排配置
│   ├── phone_camera_test.py           # 手机摄像头测试脚本
│   ├── phone_camera_yolov8n_test.py   # 手机摄像头 + YOLOv8n 测试脚本
│   ├── face-recognition-service/      # 人脸识别微服务
│   │   ├── app.py                     # Flask 应用入口
│   │   ├── face_engine.py             # 人脸识别引擎
│   │   ├── requirements.txt           # Python 依赖
│   │   └── faces/                     # 人脸数据存储
│   ├── model-inference-service/       # 模型推理微服务
│   │   ├── app.py                     # Flask 应用入口
│   │   ├── ai_agent.py               # AI 智能体
│   │   ├── chat_agent.py             # 对话智能体
│   │   ├── requirements.txt           # Python 依赖
│   │   └── ByteTrack/                # ByteTrack 目标跟踪
│   ├── user-center-service/           # 用户中心微服务
│   │   ├── app.py                     # Flask 应用入口
│   │   ├── Dockerfile                 # Docker 构建文件
│   │   ├── init.sql                   # 数据库初始化脚本
│   │   └── requirements.txt           # Python 依赖
│   └── models/                        # 预训练模型文件（被 .gitignore）
│
├── front_end/                         # 前端应用（Vue.js 3 + Element Plus）
│   ├── admin/                         # 管理端（教师/管理者）
│   ├── client/                        # 客户端（学生/家长）
│   └── terminal/                      # 终端（教室硬件采集端）
│
├── agent_demo/                        # 智能体演示
│   ├── main.py                        # 主程序入口
│   ├── demo.py                        # 演示脚本
│   ├── agents/                        # 智能体模块
│   │   ├── master_agent.py            # 主控智能体
│   │   ├── context_agent.py           # 上下文智能体
│   │   └── perception_agent.py        # 感知智能体
│   ├── data_hub/                      # 数据中心（待扩展）
│   └── database/                      # 数据库文件
│
├── bytetrack_demo/                    # ByteTrack 多目标跟踪演示
│   ├── main.py                        # 主程序
│   ├── main_old_yolo_version.py       # 旧版 YOLO 实现（备份）
│   ├── ByteTrack/                     # ByteTrack 第三方库
│   ├── yolov7/                        # YOLOv7 第三方库
│   └── yolov8n_openvino_model/        # OpenVINO 模型
│
├── llm_analysis/                      # LLM 文本分析模块
│   ├── llm_analysis.py                # 大语言模型分析脚本
│   └── prompt.txt                     # 提示词模板
│
├── mllm_demo/                         # 多模态大模型演示
│   ├── vlm_agent_demo.py             # VLM 智能体演示
│   └── frames/                        # 关键帧图像
│
├── model_training/                    # 模型训练
│   ├── train.py                       # 训练脚本
│   ├── kaggle_test.py                 # Kaggle 测试脚本
│   ├── data/                          # 训练数据
│   ├── models/                        # 模型定义
│   ├── notebooks/                     # Jupyter Notebooks
│   ├── scripts/                       # 辅助脚本
│   ├── utils/                         # 工具函数
│   └── weights/                       # 训练权重
│
├── model_testing/                     # 模型测试
│   ├── test.py                        # 测试脚本
│   └── data/                          # 测试数据
│
├── pose_demo/                         # 姿态检测演示
│   ├── pose_detection.py              # 姿态检测脚本
│   ├── crowdhuman_detection.py        # CrowdHuman 数据集检测
│   └── demo/                          # 演示视频
│
├── rag_demo/                          # RAG（检索增强生成）演示
│   ├── rag_engine.py                  # RAG 引擎
│   └── assets/                        # 资源文件
│
├── yolo_track_detection_demo/         # YOLO 跟踪检测演示
│   ├── yolo_track_demo.py             # YOLO 跟踪演示
│   └── demo/                          # 演示视频
│
├── yolov5-slowfast-deepsort-PytorchVideo/  # 第三方库（行为识别）
│   ├── yolo_slowfast.py               # YOLO + SlowFast 主程序
│   ├── yolo_custom_slowfast.py        # 自定义 SlowFast 版本
│   └── requirements.txt               # Python 依赖
│
└── docs/                              # 项目文档
    ├── 检测模块设计.md                 # 目标检测模块设计文档
    └── 报告生成模块设计.md             # 报告生成模块设计文档
```

---

## 核心功能

1. 高精度实时监测：利用多目标跟踪算法（MOT）实时感知课堂内学生的考勤状态及空间分布。
2. 多维行为语义分析：通过深度学习模型识别学生在课堂上的关键行为（如举手互动、阅读、书写等），生成量化的专注度趋势图表。
3. 闭环化课堂管理：自动生成课堂摘要报告，针对异常学习状态实现温和的预警提示，辅助教师优化教学节奏。

---

## 技术栈

| 模块 | 技术实现 |
| --- | --- |
| **后端 (Back-end)** | Flask / FastAPI |
| **前端 (Front-end)** | Vue.js 3 + Element Plus |
| **视觉引擎 (CV Core)** | YOLOv8 / ByteTrack / OpenVINO |
| **部署 (DevOps)** | Docker + Nginx |

---

## 模型部署与优化
为确保系统在边缘侧终端的实时性，本项目采用 OpenVINO 框架进行推理加速：

模型导出命令（以 YOLOv8 为例）：

```bash
# 导出 OpenVINO 格式以优化在 Intel 处理器上的推理性能
yolo export model=yolov8n.pt format=openvino imgsz=640 half=True
```

---

## 智能交互 Agent 模块
系统搭载双范式AI智能体，兼顾自动化分析与交互式问答：

1. 任务驱动型报告Agent：根据课堂行为检测结果、关键帧图像与课程信息，自动生成结构化课堂综合分析报告，用于课后批量归档与教学评估。
2. LLM 交互式对话Agent（Function Calling）：基于通义千问大模型实现原生工具调用能力，内置数据库查询、课堂详情读取两类工具。