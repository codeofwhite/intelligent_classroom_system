# Intelligent Classroom System (智能课堂系统)

本项目采用**三端分离架构**开发的智能课堂系统，涵盖后端服务、管理员管理端以及学生/教师客户端。系统旨在通过数字化手段优化课堂互动与教学管理。

## 项目架构

系统主要由以下三个部分组成：

* **`back_end`**: 提供核心业务逻辑、数据库交互及视频处理 API。
* **`front_end` (Client)**: 面向学生与家长。
* **`admin_frontend` (Docs 提及的管理员端)**: 面向老师与学校管理者，负责用户管理、课程排期及数据统计。

---

## 核心功能

---

## 技术栈

| 模块 | 技术实现 (建议补充) |
| --- | --- |
| **后端 (Back-end)** | Flask |
| **前端 (Front-end)** | Vue.js |
| **其他** | Docker |

---

转 openvino 命令
- yolov8 以上直接过
```bash
yolo export model=yolov8n.pt format=openvino imgsz=640
```