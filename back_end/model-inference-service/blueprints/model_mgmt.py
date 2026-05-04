"""
模型管理 蓝图
- 获取可用模型列表
- 切换当前模型
"""
import os
from flask import Blueprint, request, jsonify
from ultralytics import YOLO

from config import MODEL_CONFIGS, MODELS_DIR
import shared

model_bp = Blueprint("model_mgmt", __name__)


@model_bp.route('/get_models', methods=['GET'])
def get_models():
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return jsonify({"models": models, "current": shared.current_model_name})


@model_bp.route('/switch_model', methods=['POST'])
def switch_model():
    target_model = request.json.get('model_name')

    if target_model not in MODEL_CONFIGS:
        return jsonify({"status": "error", "msg": "模型未配置，请先在 MODEL_CONFIGS 中定义"}), 404

    try:
        cfg = MODEL_CONFIGS[target_model]
        model_path = os.path.join(MODELS_DIR, target_model)
        shared.model = YOLO(model_path, task=cfg['task'])

        shared.current_model_name = target_model
        shared.current_model_config = cfg

        return jsonify({
            "status": "success",
            "current": shared.current_model_name,
            "labels": cfg['labels_cn']
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500