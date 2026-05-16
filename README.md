# Intelligent Classroom System

基于计算机视觉的智能课堂行为分析系统，通过 YOLO 目标检测 + 多目标跟踪 + 大模型 Agent，实现课堂专注度实时监测、行为分析、人脸签到和智能报告生成。

## 项目结构

```
├── back_end/
│   ├── model-inference-service/   # 主后端（视频推理、AI Agent、聊天对话）
│   ├── face-recognition-service/  # 人脸识别签到服务
│   └── user-center-service/       # 用户中心（登录、权限）
├── front_end/
│   ├── admin/       # 教师管理端
│   ├── client/      # 学生/家长端
│   └── terminal/    # 教室采集端（推流+签到）
├── demos/           # 算法演示脚本
└── tests/           # 接口自动化测试
```

## 技术栈

- **后端**: Flask + MySQL + MinIO
- **前端**: Vue 3 + Element Plus
- **视觉**: YOLOv8 / ByteTrack / OpenVINO
- **AI**: 通义千问（DashScope）+ Function Calling Agent
- **部署**: Docker Compose

## 快速开始

### 1. 配置环境变量

复制模板并填入你的密钥：

```bash
cp .env.example .env
```

编辑 `.env`，填入 DashScope API Key、数据库密码、MinIO 凭据等。

### 2. 启动后端

```bash
cd back_end/model-inference-service
pip install -r requirements.txt
python app.py
```

### 3. 启动前端

```bash
cd front_end/admin
npm install
npm run dev
```

## 核心功能

- **课堂行为检测**: YOLO 实时识别举手、看书、写字、玩手机、低头、睡觉等行为
- **专注度分析**: 自动计算专注率，生成课堂报告
- **人脸签到**: 摄像头自动识别学生并签到
- **AI 助手**: 教师可对话式查询历史课堂数据、班级排行、学生表现
- **家校互通**: 家长端查看孩子课堂表现与 AI 评语

## Docker 部署

```bash
cd back_end
docker-compose up -d
```

## 注意事项

- `.env` 文件包含敏感密钥，**不要提交到 Git**
- `.puml`、`.tex`、`.bib` 文件为论文相关，已在 `.gitignore` 中排除
- 模型权重文件（`.pt`、`.onnx`）不提交，请自行下载或训练