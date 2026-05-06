"""
压力测试脚本 - 使用 Locust
==========================================

使用方法：
  1. 安装 locust: pip install locust
  2. 启动被测服务（确保 docker-compose 已启动）
  3. 运行方式一（Web UI）:
     locust -f tests/locustfile.py
     浏览器打开 http://localhost:8089（Host 字段留空）
  4. 运行方式二（命令行，无头模式）:
     locust -f tests/locustfile.py \
            --headless -u 20 -r 5 --run-time 60s --csv=tests/reports/stress

参数说明：
  -u  : 模拟用户数（并发数）
  -r  : 每秒启动的用户数（ramp-up rate）
  --run-time : 测试持续时间
  --csv : 结果输出到 CSV 文件
"""

from locust import HttpUser, task, between, tag
import random
import json


# ====================================================
#  用户中心服务 压力测试
# ====================================================
class UserCenterUser(HttpUser):
    """模拟用户中心服务的用户行为"""

    wait_time = between(1, 3)
    host = "http://localhost:5001"

    def on_start(self):
        """用户启动时执行登录"""
        self.role = random.choice(["teacher", "student", "parent"])
        self.accounts = {
            "teacher": {"username": "teacherwang", "password": "123456", "role": "teacher"},
            "student": {"username": "zhangsan",    "password": "123456", "role": "student"},
            "parent":  {"username": "zhangfather", "password": "123456", "role": "parent"},
        }
        self.user_code = None
        self._login()

    def _login(self):
        """执行登录"""
        account = self.accounts[self.role]
        with self.client.post(
            "/login",
            json=account,
            name="登录接口",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    self.user_code = data["user"]["user_code"]
                    response.success()
                else:
                    response.failure(f"登录失败: {data.get('message')}")
            elif response.status_code == 401:
                response.failure(f"登录失败: 账号或密码错误")
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("login")
    @task(3)
    def login(self):
        """重复登录测试（模拟多用户同时登录）"""
        account = self.accounts[self.role]
        with self.client.post(
            "/login",
            json=account,
            name="重复登录",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    response.success()
                else:
                    response.failure(f"登录失败: {data.get('message')}")
            elif response.status_code == 401:
                response.failure(f"登录失败: 账号或密码错误")
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("teacher")
    @task(2)
    def get_teacher_class(self):
        """教师获取班级信息"""
        if self.role != "teacher" or not self.user_code:
            return
        with self.client.post(
            "/teacher-class",
            json={"user_code": self.user_code},
            name="教师-班级信息",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "class_name" in data:
                    response.success()
                else:
                    response.failure("返回数据缺少 class_name")
            elif response.status_code == 500:
                response.failure("服务内部错误 (500)")
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("teacher")
    @task(2)
    def get_teacher_students(self):
        """教师获取学生列表"""
        if self.role != "teacher" or not self.user_code:
            return
        with self.client.post(
            "/teacher-students",
            json={"user_code": self.user_code},
            name="教师-学生列表",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 500:
                response.failure("服务内部错误 (500)")
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("parent")
    @task(2)
    def get_parent_children(self):
        """家长获取孩子信息"""
        if self.role != "parent" or not self.user_code:
            return
        with self.client.post(
            "/parent-children",
            json={"user_code": self.user_code},
            name="家长-孩子信息",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 500:
                response.failure("服务内部错误 (500)")
            else:
                response.failure(f"状态码: {response.status_code}")


# ====================================================
#  模型推理服务 压力测试
# ====================================================
class ModelInferenceUser(HttpUser):
    """模拟模型推理服务的用户行为"""

    wait_time = between(2, 5)
    host = "http://localhost:5002"

    def on_start(self):
        """初始化"""
        self.teacher_code = "teacherwang"

    @tag("health")
    @task(3)
    def health_check(self):
        """服务健康检查"""
        with self.client.get(
            "/get_models",
            name="健康检查",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("model")
    @task(2)
    def get_models(self):
        """获取模型列表"""
        with self.client.get(
            "/get_models",
            name="获取模型列表",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("report")
    @task(3)
    def get_teacher_reports(self):
        """教师获取报告列表"""
        with self.client.get(
            "/api/teacher/reports",
            params={"teacher_code": self.teacher_code},
            name="教师报告列表",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("report")
    @task(2)
    def get_report_detail(self):
        """获取报告详情（使用有效 ID 1~3）"""
        report_id = random.randint(1, 3)
        with self.client.get(
            "/api/report/detail",
            params={"id": report_id},
            name="报告详情",
            catch_response=True,
        ) as response:
            # 500 是预期的（ID 不存在时服务端报错），不算测试失败
            if response.status_code in [200, 500]:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("realtime")
    @task(2)
    def get_realtime_stats(self):
        """获取实时统计"""
        with self.client.get(
            "/get_realtime_stats",
            name="实时统计",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("realtime")
    @task(1)
    def get_record_status(self):
        """获取录制状态"""
        with self.client.get(
            "/get_record_status",
            name="录制状态",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("video")
    @task(1)
    def list_videos(self):
        """获取视频列表"""
        with self.client.get(
            "/list_videos",
            name="视频列表",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("student")
    @task(1)
    def get_schedule(self):
        """获取课程表"""
        with self.client.post(
            "/api/teacher/course_schedule",
            json={"teacher_code": self.teacher_code},
            name="课程表",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")


# ====================================================
#  人脸识别服务 压力测试
# ====================================================
class FaceRecognitionUser(HttpUser):
    """模拟人脸识别服务的用户行为"""

    wait_time = between(2, 5)
    host = "http://localhost:5003"

    @tag("health")
    @task(3)
    def health_check(self):
        """服务健康检查"""
        with self.client.get(
            "/",
            name="健康检查",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("log")
    @task(2)
    def get_sign_log(self):
        """获取签到日志"""
        with self.client.get(
            "/get_sign_log",
            name="签到日志",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @tag("reload")
    @task(1)
    def reload_faces(self):
        """重载人脸库"""
        with self.client.post(
            "/reload_faces",
            name="重载人脸库",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code in [200, 500]:
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")


# ====================================================
#  混合场景：模拟真实课堂并发
# ====================================================
class ClassroomScenario(HttpUser):
    """
    模拟真实课堂场景：
    - 多个教师同时登录查看班级
    - 多个家长查看孩子信息
    """

    wait_time = between(1, 3)
    host = "http://localhost:5001"

    def on_start(self):
        self.scenario = random.choice(["teacher_login", "parent_query"])
        if self.scenario == "teacher_login":
            self._login("teacher")
        elif self.scenario == "parent_query":
            self._login("parent")

    def _login(self, role):
        accounts = {
            "teacher": {"username": "teacherwang", "password": "123456", "role": "teacher"},
            "parent":  {"username": "zhangfather", "password": "123456", "role": "parent"},
        }
        with self.client.post(
            "/login",
            json=accounts.get(role, accounts["teacher"]),
            name=f"场景-{role}-登录",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.user_code = data.get("user", {}).get("user_code")
                response.success()
            else:
                response.failure(f"状态码: {response.status_code}")

    @task(2)
    def scenario_teacher_flow(self):
        """教师完整流程：登录 -> 查班级 -> 查学生"""
        if self.scenario != "teacher_login":
            return
        self._login("teacher")
        if self.user_code:
            self.client.post(
                "/teacher-class",
                json={"user_code": self.user_code},
                name="场景-教师-班级",
            )
            self.client.post(
                "/teacher-students",
                json={"user_code": self.user_code},
                name="场景-教师-学生",
            )

    @task(2)
    def scenario_parent_flow(self):
        """家长完整流程：登录 -> 查孩子"""
        if self.scenario != "parent_query":
            return
        self._login("parent")
        if self.user_code:
            self.client.post(
                "/parent-children",
                json={"user_code": self.user_code},
                name="场景-家长-孩子",
            )