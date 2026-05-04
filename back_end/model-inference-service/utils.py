"""
工具函数模块 - 通用辅助函数
"""


def get_color(category_id):
    """根据类别ID返回绘制颜色"""
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return colors[category_id % len(colors)]


def round0(v):
    """安全地将值转为四舍五入的整数"""
    return round(float(v or 0))