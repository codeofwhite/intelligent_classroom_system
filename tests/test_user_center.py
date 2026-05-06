"""
用户中心服务 - API 接口测试
测试范围：
  1. 登录接口（多角色）
  2. 教师端接口
  3. 家长端接口
"""
import pytest
import requests


# ====================================================
#  登录接口测试
# ====================================================
class TestLogin:
    """登录接口 /login 测试"""

    def test_login_teacher_success(self, user_center_url):
        """教师登录 - 正确账号密码"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "teacherwang", "password": "123456", "role": "teacher"},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["user"]["role"] == "teacher"
        assert "user_code" in data["user"]

    def test_login_student_success(self, user_center_url):
        """学生登录 - 正确账号密码"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "zhangsan", "password": "123456", "role": "student"},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["user"]["role"] == "student"

    def test_login_parent_success(self, user_center_url):
        """家长登录 - 正确账号密码"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "zhangfather", "password": "123456", "role": "parent"},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["user"]["role"] == "parent"

    def test_login_wrong_password(self, user_center_url):
        """登录失败 - 错误密码"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "T001", "password": "wrong", "role": "teacher"},
            timeout=5,
        )
        assert r.status_code == 401
        data = r.json()
        assert data["status"] == "error"

    def test_login_missing_params(self, user_center_url):
        """登录失败 - 缺少参数"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "T001"},
            timeout=5,
        )
        assert r.status_code == 400
        data = r.json()
        assert data["status"] == "error"

    def test_login_empty_body(self, user_center_url):
        """登录失败 - 空请求体"""
        r = requests.post(
            f"{user_center_url}/login",
            json={},
            timeout=5,
        )
        assert r.status_code == 400

    def test_login_nonexistent_user(self, user_center_url):
        """登录失败 - 不存在的用户"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "notexist", "password": "123456", "role": "teacher"},
            timeout=5,
        )
        assert r.status_code == 401

    def test_login_wrong_role(self, user_center_url):
        """登录失败 - 角色不匹配"""
        r = requests.post(
            f"{user_center_url}/login",
            json={"username": "teacherwang", "password": "123456", "role": "student"},
            timeout=5,
        )
        assert r.status_code == 401


# ====================================================
#  教师端接口测试
# ====================================================
class TestTeacherAPI:
    """教师端接口测试"""

    def test_teacher_class_success(self, user_center_url, teacher_token):
        """获取教师班级信息"""
        user_code = teacher_token["user"]["user_code"]
        r = requests.post(
            f"{user_center_url}/teacher-class",
            json={"user_code": user_code},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert "class_name" in data
        assert "class_code" in data
        assert "students" in data
        assert "student_count" in data
        assert isinstance(data["students"], list)

    def test_teacher_class_invalid_code(self, user_center_url):
        """教师班级 - 无效 user_code"""
        r = requests.post(
            f"{user_center_url}/teacher-class",
            json={"user_code": "INVALID_CODE"},
            timeout=5,
        )
        # 服务端可能返回 500 或空结果（取决于实现）
        assert r.status_code in [200, 500]

    def test_teacher_students_success(self, user_center_url, teacher_token):
        """获取教师学生列表"""
        user_code = teacher_token["user"]["user_code"]
        r = requests.post(
            f"{user_center_url}/teacher-students",
            json={"user_code": user_code},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert "class_name" in data
        assert "students" in data
        assert isinstance(data["students"], list)
        # 检查学生字段
        if len(data["students"]) > 0:
            student = data["students"][0]
            assert "student_code" in student or "student_name" in student

    def test_teacher_students_empty_code(self, user_center_url):
        """教师学生 - 空 user_code"""
        r = requests.post(
            f"{user_center_url}/teacher-students",
            json={"user_code": ""},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["students"] == []


# ====================================================
#  家长端接口测试
# ====================================================
class TestParentAPI:
    """家长端接口测试"""

    def test_parent_children_success(self, user_center_url, parent_token):
        """获取家长绑定的孩子信息"""
        user_code = parent_token["user"]["user_code"]
        r = requests.post(
            f"{user_center_url}/parent-children",
            json={"user_code": user_code},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert "children" in data
        assert isinstance(data["children"], list)

    def test_parent_children_invalid_code(self, user_center_url):
        """家长获取孩子 - 无效 user_code"""
        r = requests.post(
            f"{user_center_url}/parent-children",
            json={"user_code": "INVALID"},
            timeout=5,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["children"] == []