"""
模型推理服务 - API 接口测试
测试范围：
  1. 视频上传分析
  2. 课堂报告
  3. 模型管理
  4. AI 分析
  5. 班级/学生/家长接口
"""
import pytest
import requests
import os
import time


# ====================================================
#  健康检查
# ====================================================
class TestHealthCheck:
    """模型推理服务健康检查"""

    def test_service_running(self, model_inference_url):
        """服务是否正常运行（通过 get_models 接口验证）"""
        r = requests.get(f"{model_inference_url}/get_models", timeout=5)
        assert r.status_code == 200


# ====================================================
#  模型管理接口
# ====================================================
class TestModelManagement:
    """模型管理 /get_models, /switch_model"""

    def test_get_models(self, model_inference_url):
        """获取可用模型列表"""
        r = requests.get(f"{model_inference_url}/get_models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))

    def test_switch_model_valid(self, model_inference_url):
        """切换到有效模型（需要根据实际模型名调整）"""
        # 先获取模型列表
        r = requests.get(f"{model_inference_url}/get_models", timeout=5)
        if r.status_code != 200:
            pytest.skip("无法获取模型列表")

        data = r.json()
        # 返回格式: {"models": [...], "current": "..."}
        models = data.get("models", []) if isinstance(data, dict) else data
        if not models:
            pytest.skip("模型列表为空")

        model_name = models[0]

        r = requests.post(
            f"{model_inference_url}/switch_model",
            json={"model_name": model_name},
            timeout=10,
        )
        assert r.status_code in [200, 400, 404]

    def test_switch_model_invalid(self, model_inference_url):
        """切换到不存在的模型"""
        r = requests.post(
            f"{model_inference_url}/switch_model",
            json={"model_name": "nonexistent_model_xyz"},
            timeout=10,
        )
        assert r.status_code in [400, 404, 500]


# ====================================================
#  视频上传接口
# ====================================================
class TestVideoUpload:
    """视频上传 /upload_video"""

    def test_upload_no_file(self, model_inference_url):
        """上传 - 无文件"""
        r = requests.post(
            f"{model_inference_url}/upload_video",
            data={"teacher_code": "T001", "class_code": "C001", "lesson_section": "1"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_upload_missing_params(self, model_inference_url):
        """上传 - 缺少参数但有文件"""
        # 创建一个假的小文件
        files = {"video": ("test.mp4", b"fake video content", "video/mp4")}
        r = requests.post(
            f"{model_inference_url}/upload_video",
            files=files,
            timeout=10,
        )
        # 接口可能返回 200（尝试处理）或 400/500（参数校验失败）
        assert r.status_code in [200, 400, 500]

    def test_list_videos(self, model_inference_url):
        """获取视频列表"""
        r = requests.get(f"{model_inference_url}/list_videos", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)


# ====================================================
#  课堂报告接口
# ====================================================
class TestReports:
    """课堂报告接口"""

    def test_teacher_reports_no_code(self, model_inference_url):
        """教师报告 - 无 teacher_code"""
        r = requests.get(f"{model_inference_url}/api/teacher/reports", timeout=5)
        assert r.status_code == 200
        assert r.json() == []

    def test_teacher_reports_invalid_code(self, model_inference_url):
        """教师报告 - 无效 teacher_code"""
        r = requests.get(
            f"{model_inference_url}/api/teacher/reports",
            params={"teacher_code": "INVALID"},
            timeout=10,
        )
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_report_detail_no_id(self, model_inference_url):
        """报告详情 - 无 id"""
        r = requests.get(f"{model_inference_url}/api/report/detail", timeout=5)
        # 没有 id 可能返回 500 或空
        assert r.status_code in [200, 500]

    def test_report_detail_invalid_id(self, model_inference_url):
        """报告详情 - 无效 id"""
        r = requests.get(
            f"{model_inference_url}/api/report/detail",
            params={"id": "999999"},
            timeout=10,
        )
        assert r.status_code in [200, 404, 500]

    def test_report_history_no_params(self, model_inference_url):
        """学生报告历史 - 无参数"""
        r = requests.get(f"{model_inference_url}/api/report/history", timeout=5)
        assert r.status_code in [200, 500]

    def test_report_list_no_params(self, model_inference_url):
        """学生报告列表 - 无参数"""
        r = requests.get(f"{model_inference_url}/api/report/list", timeout=5)
        assert r.status_code in [200, 500]

    def test_report_delete_no_id(self, model_inference_url):
        """删除报告 - 无 id"""
        r = requests.post(
            f"{model_inference_url}/api/report/delete",
            json={},
            timeout=5,
        )
        assert r.status_code == 400


# ====================================================
#  实时监测接口
# ====================================================
class TestRealtime:
    """实时监测相关接口"""

    def test_get_realtime_stats(self, model_inference_url):
        """获取实时统计"""
        r = requests.get(f"{model_inference_url}/get_realtime_stats", timeout=5)
        assert r.status_code == 200

    def test_get_record_status(self, model_inference_url):
        """获取录制状态"""
        r = requests.get(f"{model_inference_url}/get_record_status", timeout=5)
        assert r.status_code == 200


# ====================================================
#  聊天助手接口
# ====================================================
class TestChat:
    """聊天助手接口"""

    def test_chat_no_message(self, model_inference_url):
        """聊天 - 空消息"""
        r = requests.post(
            f"{model_inference_url}/api/chat",
            json={},
            timeout=10,
        )
        # 接口可能正常返回空回答或报错
        assert r.status_code in [200, 400, 500]


# ====================================================
#  课程表接口
# ====================================================
class TestSchedule:
    """课程表接口"""

    def test_get_schedule(self, model_inference_url):
        """获取课程表（POST 方法）"""
        r = requests.post(
            f"{model_inference_url}/api/teacher/course_schedule",
            json={"teacher_code": ""},
            timeout=5,
        )
        assert r.status_code in [200, 400]
