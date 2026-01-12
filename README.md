# Intelligent Classroom System (智能课堂系统)

本项目是一个采用**三端分离架构**开发的智能课堂系统，涵盖了后端服务、管理员管理端以及学生/教师客户端。系统旨在通过数字化手段优化课堂互动与教学管理。

## 🚀 项目架构

系统主要由以下三个部分组成：

* **`back_end`**: 提供核心业务逻辑、数据库交互及视频处理 API。
* **`front_end` (Client)**: 面向学生与教师的移动端或 Web 应用，侧重于课堂互动与视频学习。
* **`admin_frontend` (Docs 提及的管理员端)**: 面向系统管理员，负责用户管理、课程排期及数据统计。

---

## ✨ 核心功能

* **三端分离**: 采用标准的 RESTful API 交互，保障各端独立开发与部署。
* **视频 API 集成**: 支持视频资源的上传、流媒体播放及相关的教学视频管理。
* **权限管理**: 针对管理员、教师和学生设计了完备的角色访问控制（RBAC）。
* **智能教学辅助**: (可根据你的具体功能补充，如：自动考勤、实时问答等)。

---

## 🛠️ 技术栈

| 模块 | 技术实现 (建议补充) |
| --- | --- |
| **后端 (Back-end)** | Java / Python / Node.js + MySQL / Redis |
| **客户端 (Front-end)** | Vue.js / React / Flutter |
| **管理端 (Admin)** | Ant Design Pro / Element Plus |
| **其他** | Docker, Nginx, ffmpeg (视频处理) |

---

## 📂 目录结构

```text
intelligent_classroom_system/
├── back_end/          # 后端源代码
├── front_end/         # 客户端前端代码
├── docs/              # 项目文档、API 接口定义、数据库模型
└── README.md          # 项目自述文件

```

---

## ⚙️ 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/codeofwhite/intelligent_classroom_system.git
cd intelligent_classroom_system

```

### 2. 环境配置

* **后端**: 进入 `back_end` 目录，配置数据库连接并运行。
* **前端**: 进入对应的 `front_end` 目录，执行 `npm install`。

### 3. API 测试

目前后端视频 API 正在测试中，相关接口文档可参考 `docs` 文件夹。

---

## 📝 最近更新

* **优化前端设置**: 提升了 UI 交互体验。
* **后端视频 API 测试**: 完成了基础视频流接口的跑通。

---

## 🤝 贡献指南

欢迎提交 Issue 或 Pull Request。在提交代码前，请确保已在本地通过基本的功能测试。

---