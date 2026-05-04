"""
AI 分析 蓝图
- AI 生成学生行为评语
- 家校共育 AI 综合分析建议
"""
import pymysql
from flask import Blueprint, request, jsonify
from openai import OpenAI

from shared import get_db_connection

ai_bp = Blueprint("ai", __name__)


@ai_bp.route("/api/ai/analyze", methods=["POST"])
def ai_analyze():
    data = request.json
    student_code = data.get("student_code")
    normal = data.get("normal_posture")
    raised = data.get("raised_hand")
    down = data.get("looking_down")
    focus = data.get("focus_rate")

    prompt = f"""
你是小学/中学课堂行为分析师，请用温和、鼓励、专业的语气写一段评语。
行为数据：
正常坐姿：{normal}
举手次数：{raised}
低头次数：{down}
专注度：{focus}%

要求：80字左右，适合家长阅读。
"""

    client = OpenAI(
        api_key="sk-06abd7a7eb514b3ebd611412f0dc3531",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    res = completion.choices[0].message.content
    return jsonify({"comment": res})


@ai_bp.route("/api/ai/advice", methods=["GET"])
def ai_advice():
    student_code = request.args.get("student_code")
    if not student_code:
        return jsonify({"summary": "", "advice": ""})

    try:
        db_tmp = get_db_connection()
        cursor = db_tmp.cursor(pymysql.cursors.DictCursor)

        cursor.execute("""
            SELECT focus_rate, normal_posture, raised_hand, looking_down, ai_comment
            FROM student_reports
            WHERE student_code=%s
            ORDER BY lesson_time DESC
        """, (student_code,))
        reports = cursor.fetchall()

        if not reports:
            cursor.close()
            db_tmp.close()
            return jsonify({
                "summary": "暂无历史课堂数据，无法生成分析",
                "advice": "请等待课堂数据生成后再查看"
            })

        # 统计
        total = len(reports)
        avg_focus = round(sum(r["focus_rate"] for r in reports) / total)
        good_posture = sum(r["normal_posture"] for r in reports)
        total_hand = sum(r["raised_hand"] for r in reports)
        total_down = sum(r["looking_down"] for r in reports)

        summary = f"""
该生近 {total} 节课平均专注度 {avg_focus}%，整体课堂状态良好。
累计坐姿达标 {good_posture} 次，主动举手发言 {total_hand} 次，分心低头 {total_down} 次。
        """.strip()

        advice = f"""
【家校共育建议】
1. 该生专注度表现{"优秀" if avg_focus >= 90 else "良好" if avg_focus >= 80 else "一般"}，建议继续保持专注习惯。
2. 主动发言积极性{"很高" if total_hand >= total * 3 else "一般"}，建议多鼓励课堂参与。
3. 分心情况{"较少" if total_down <= total * 1 else "偏多"}，家校共同引导注意力管理。
4. 家庭配合：规律作息、减少电子产品干扰，与学校同步培养学习习惯。
        """.strip()

        cursor.close()
        db_tmp.close()

        return jsonify({
            "summary": summary,
            "advice": advice
        })

    except Exception as e:
        print("AI分析错误:", e)
        return jsonify({"summary": "", "advice": ""})