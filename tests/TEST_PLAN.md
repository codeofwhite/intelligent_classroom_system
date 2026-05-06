# 智能课堂系统 - 测试方案

## 目录

1. [测试概述](#1-测试概述)
2. [测试环境准备](#2-测试环境准备)
3. [功能测试（API 接口测试）](#3-功能测试)
4. [压力测试（Locust）](#4-压力测试)
5. [测试执行指南](#5-测试执行指南)
6. [测试指标与通过标准](#6-测试指标与通过标准)

---

## 1. 测试概述

### 1.1 测试类型

| 测试类型 | 说明 | 工具 |
|---------|------|------|
| **功能测试** | 验证每个 API 接口的功能正确性 | pytest + requests |
| **接口测试** | 测试参数校验、边界条件、异常处理 | pytest |
| **压力测试** | 模拟高并发用户访问，测试系统承载能力 | Locust |
| **场景测试** | 模拟真实课堂场景的完整业务流程 | Locust |

### 1.2 被测服务

| 服务 | 端口 | 说明 |
|------|------|------|
| user-center-service | 5001 | 用户中心（登录、教师、家长） |
| model-inference-service | 5002 | 模型推理（视频分析、报告、AI） |
| face-recognition-service | 5003 | 人脸识别签到 |

---

## 2. 测试环境准备

### 2.1 安装测试依赖

```bash
pip install -r tests/requirements-test.txt
```

### 2.2 启动被测服务

```bash
# 方式一：使用 Docker Compose 启动数据库等基础设施
cd back_end
docker-compose up -d

# 方式二：手动启动各个服务（开发模式）
# 终端1 - 用户中心服务
cd back_end/user-center-service && python app.py

# 终端2 - 模型推理服务
cd back_end/model-inference-service && python app.py

# 终端3 - 人脸识别服务
cd back_end/face-recognition-service && python app.py
```

### 2.3 准备测试数据

确保数据库中存在测试账号：
- 教师账号：`teacherwang / 123456`（role=teacher）
- 学生账号：`zhangsan / 123456`（role=student）
- 家长账号：`zhangfather / 123456`（role=parent）

> **注意**：如果测试账号的用户名/密码不同，请修改 `tests/conftest.py` 中的 `TEST_ACCOUNTS`。

---

## 3. 功能测试

### 3.1 测试文件结构

```
tests/
├── conftest.py                 # 公共配置、fixtures
├── test_user_center.py         # 用户中心服务测试
│   ├── TestLogin               # 登录接口（8个用例）
│   ├── TestTeacherAPI          # 教师接口（4个用例）
│   └── TestParentAPI           # 家长接口（2个用例）
├── test_model_inference.py     # 模型推理服务测试
│   ├── TestHealthCheck         # 健康检查（1个用例）
│   ├── TestModelManagement     # 模型管理（3个用例）
│   ├── TestVideoUpload         # 视频上传（3个用例）
│   ├── TestReports             # 课堂报告（7个用例）
│   ├── TestRealtime            # 实时监测（2个用例）
│   ├── TestChat                # 聊天助手（1个用例）
│   └── TestSchedule            # 课程表（1个用例）
├── test_face_recognition.py    # 人脸识别服务测试
│   ├── TestFaceServiceHealth   # 健康检查（2个用例）
│   ├── TestSignLog             # 签到日志（2个用例）
│   ├── TestReloadFaces         # 人脸库重载（2个用例）
│   └── TestVideoFeed           # 视频流（1个用例）
```

### 3.2 测试用例详情

#### 用户中心服务测试用例

| 用例 | 接口 | 测试内容 | 预期结果 |
|------|------|---------|---------|
| 教师登录 | POST /login | 正确账号密码 | 200, role=teacher |
| 学生登录 | POST /login | 正确账号密码 | 200, role=student |
| 家长登录 | POST /login | 正确账号密码 | 200, role=parent |
| 错误密码 | POST /login | 密码错误 | 401 |
| 缺少参数 | POST /login | 只传 username | 400 |
| 空请求体 | POST /login | {} | 400 |
| 不存在用户 | POST /login | 不存在的用户名 | 401 |
| 角色不匹配 | POST /login | teacher 用 student 角色 | 401 |
| 教师班级 | POST /teacher-class | 有效 user_code | 200, 含班级信息 |
| 教师学生 | POST /teacher-students | 有效 user_code | 200, 含学生列表 |
| 家长孩子 | POST /parent-children | 有效 user_code | 200, 含 children |

---

## 4. 压力测试

### 4.1 压力测试场景

| 场景 | 用户类 | 模拟行为 | 任务权重 |
|------|--------|---------|---------|
| 用户中心 | UserCenterUser | 登录、查班级、查学生 | 登录3: 教师2: 家长2 |
| 模型推理 | ModelInferenceUser | 查报告、查模型、实时统计 | 报告3: 模型2: 实时2 |
| 人脸识别 | FaceRecognitionUser | 查日志、重载人脸库 | 日志3: 重载1 |
| 课堂混合 | ClassroomScenario | 教师+家长完整业务流程 | 各2 |

### 4.2 压力测试参数建议

| 测试阶段 | 并发用户 | 启动速率 | 持续时间 | 目的 |
|---------|---------|---------|---------|------|
| 冒烟 | 5 | 1 | 30s | 快速验证系统可用 |
| 轻负载 | 20 | 5 | 60s | 正常负载测试 |
| 中负载 | 50 | 10 | 120s | 模拟高峰时段 |
| 重负载 | 100 | 20 | 180s | 极限承载测试 |
| 稳定性 | 30 | 5 | 600s | 长时间稳定性验证 |

### 4.3 关注指标

| 指标 | 说明 | 参考阈值 |
|------|------|---------|
| **RPS** (Requests Per Second) | 每秒处理请求数 | ≥ 50 |
| **Avg Response Time** | 平均响应时间 | ≤ 500ms |
| **P95 Response Time** | 95% 请求的响应时间 | ≤ 1000ms |
| **P99 Response Time** | 99% 请求的响应时间 | ≤ 2000ms |
| **Failure Rate** | 请求失败率 | ≤ 1% |
| **峰值并发** | 系统能承受的最大并发数 | ≥ 50 |

---

## 5. 测试执行指南

### 5.1 运行功能测试

```bash
# 运行全部功能测试
pytest tests/test_user_center.py tests/test_model_inference.py tests/test_face_recognition.py -v

# 只运行某个服务的测试
pytest tests/test_user_center.py -v

# 只运行某个类的测试
pytest tests/test_user_center.py::TestLogin -v

# 运行并生成 HTML 报告
pytest tests/ -v --html=tests/reports/report.html --self-contained-html

# 运行并生成覆盖率报告
pytest tests/ -v --cov=back_end --cov-report=html
```

### 5.2 运行压力测试

```bash
# 方式一：Web UI 模式（推荐初次使用）
locust -f tests/locustfile.py --host http://localhost:5001
# 浏览器打开 http://localhost:8089

# 方式二：命令行无头模式
# 测试用户中心服务（50并发，60秒）
locust -f tests/locustfile.py \
       --host http://localhost:5001 \
       --headless \
       -u 50 -r 10 \
       --run-time 60s \
       --csv=tests/reports/stress_user_center

# 方式三：使用运行脚本
python tests/run_tests.py                     # 功能测试
python tests/run_tests.py --report             # 功能测试 + HTML 报告
python tests/run_tests.py --stress             # 压力测试（Web UI）
python tests/run_tests.py --stress-headless    # 压力测试（无头）
python tests/run_tests.py --all                # 全部测试
```

### 5.3 压力测试 Web UI 操作步骤

1. 启动 Locust：`locust -f tests/locustfile.py --host http://localhost:5001`
2. 浏览器打开 `http://localhost:8089`
3. 填写参数：
   - Number of users to simulate: `50`
   - Ramp up (users started/second): `10`
   - Host: `http://localhost:5001`
4. 点击 "Start swarming"
5. 观察实时图表：
   - **Charts** 标签页：查看 RPS、响应时间曲线
   - **Statistics** 标签页：查看各接口的详细统计
   - **Failures** 标签页：查看失败请求
6. 测试完成后下载报告

---

## 6. 测试指标与通过标准

### 6.1 功能测试通过标准

- ✅ 所有正常流程测试用例通过
- ✅ 异常输入能正确返回错误码（400/401/404）
- ✅ 不出现未捕获的 500 错误（已知的除外）

### 6.2 压力测试通过标准

- ✅ 50 并发用户下，平均响应时间 < 500ms
- ✅ 50 并发用户下，错误率 < 1%
- ✅ 服务在测试期间无崩溃
- ✅ CPU 和内存使用在合理范围内

### 6.3 测试报告样例

压力测试完成后会在 `tests/reports/` 目录生成：
- `stress_xxx_stats.csv` - 统计数据
- `stress_xxx_stats_history.csv` - 历史数据
- `stress_xxx_failures.csv` - 失败记录

可以用 Excel 打开这些 CSV 文件进行分析。

---

## 附录：常见问题

### Q1: 测试连接不上服务怎么办？
确认服务已启动，端口正确。可以用 curl 测试：
```bash
curl http://localhost:5001/
curl http://localhost:5002/
curl http://localhost:5003/
```

### Q2: 登录测试失败？
检查 `tests/conftest.py` 中的 `TEST_ACCOUNTS` 是否与数据库中的测试账号一致。

### Q3: 压力测试报错 "Connection refused"？
确保目标服务正在运行，且 Locust 的 `--host` 参数指向正确的服务地址。

### Q4: 如何只测试某个接口的压力？
在 Locust Web UI 中，可以通过 tag 过滤：
- `login` - 登录接口
- `teacher` - 教师接口
- `parent` - 家长接口
- `report` - 报告接口
- `realtime` - 实时监测接口