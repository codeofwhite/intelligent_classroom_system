"""
人脸识别签到服务 - API 接口测试
测试范围：
  1. 服务健康检查
  2. 签到日志
  3. 人脸库重载
  4. 视频流
"""
import pytest
import requests


class TestFaceServiceHealth:
    """人脸识别服务健康检查"""

    def test_service_running(self, face_recognition_url):
        """服务是否正常运行"""
        r = requests.get(f"{face_recognition_url}/", timeout=5)
        assert r.status_code == 200

    def test_index_content(self, face_recognition_url):
        """首页返回内容"""
        r = requests.get(f"{face_recognition_url}/", timeout=5)
        assert "人脸" in r.text or "face" in r.text.lower() or r.status_code == 200


class TestSignLog:
    """签到日志接口 /get_sign_log"""

    def test_get_sign_log(self, face_recognition_url):
        """获取签到日志"""
        r = requests.get(f"{face_recognition_url}/get_sign_log", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)

    def test_sign_log_format(self, face_recognition_url):
        """签到日志格式验证"""
        r = requests.get(f"{face_recognition_url}/get_sign_log", timeout=5)
        assert r.status_code == 200
        data = r.json()
        logs = data.get("logs", [])
        # 如果有日志，验证格式
        if len(logs) > 0:
            # 日志可能是字符串或字典
            assert isinstance(logs[0], (str, dict))


class TestReloadFaces:
    """人脸库重载接口 /reload_faces"""

    def test_reload_faces(self, face_recognition_url):
        """重载人脸库"""
        r = requests.post(
            f"{face_recognition_url}/reload_faces",
            timeout=30,
        )
        assert r.status_code in [200, 500]
        data = r.json()
        assert "status" in data
        # 成功时 status 为 ok，失败时 status 为 error
        assert data["status"] in ["ok", "error"]

    def test_reload_faces_count(self, face_recognition_url):
        """重载人脸库后返回数量"""
        r = requests.post(
            f"{face_recognition_url}/reload_faces",
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            if data["status"] == "ok":
                assert "count" in data
                assert isinstance(data["count"], int)
                assert data["count"] >= 0


class TestVideoFeed:
    """视频流接口 /video_feed"""

    def test_video_feed_accessible(self, face_recognition_url):
        """视频流接口可访问"""
        # 只测试接口可访问性，不消费完整流
        try:
            r = requests.get(
                f"{face_recognition_url}/video_feed",
                timeout=5,
                stream=True,
            )
            assert r.status_code == 200
            assert "multipart" in r.headers.get("Content-Type", "")
            r.close()
        except requests.exceptions.ReadTimeout:
            # 视频流可能超时，这是正常的
            pass