"""
全局测试配置与公共 fixtures
"""
import pytest
import requests
import time

# ========== 服务地址配置 ==========
BASE_URLS = {
    "user_center": "http://localhost:5001",
    "model_inference": "http://localhost:5002",
    "face_recognition": "http://localhost:5003",
}

# 测试账号（与数据库中一致）
TEST_ACCOUNTS = {
    "teacher": {"username": "teacherwang", "password": "123456", "role": "teacher"},
    "student": {"username": "zhangsan",    "password": "123456", "role": "student"},
    "parent":  {"username": "zhangfather", "password": "123456", "role": "parent"},
}


def wait_for_service(url, timeout=30):
    """等待服务启动"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="session")
def base_urls():
    """返回所有服务的 base URL"""
    return BASE_URLS


@pytest.fixture(scope="session")
def user_center_url():
    return BASE_URLS["user_center"]


@pytest.fixture(scope="session")
def model_inference_url():
    return BASE_URLS["model_inference"]


@pytest.fixture(scope="session")
def face_recognition_url():
    return BASE_URLS["face_recognition"]


@pytest.fixture(scope="session")
def teacher_token(user_center_url):
    """获取教师登录 token / 用户信息"""
    r = requests.post(
        f"{user_center_url}/login",
        json=TEST_ACCOUNTS["teacher"],
        timeout=5,
    )
    assert r.status_code == 200, f"教师登录失败: {r.text}"
    data = r.json()
    return data


@pytest.fixture(scope="session")
def student_token(user_center_url):
    """获取学生登录信息"""
    r = requests.post(
        f"{user_center_url}/login",
        json=TEST_ACCOUNTS["student"],
        timeout=5,
    )
    assert r.status_code == 200, f"学生登录失败: {r.text}"
    return r.json()


@pytest.fixture(scope="session")
def parent_token(user_center_url):
    """获取家长登录信息"""
    r = requests.post(
        f"{user_center_url}/login",
        json=TEST_ACCOUNTS["parent"],
        timeout=5,
    )
    assert r.status_code == 200, f"家长登录失败: {r.text}"
    return r.json()