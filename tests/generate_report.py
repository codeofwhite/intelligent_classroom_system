"""
测试报告生成器
==========================================
读取功能测试和压力测试结果，生成综合 HTML 报告

用法：
  python tests/generate_report.py                        # 生成报告
  python tests/generate_report.py --stress-csv tests/reports/stress_20260506  # 指定压力测试 CSV
"""
import os
import sys
import csv
import glob
from datetime import datetime


def read_csv_auto_encoding(filepath):
    """自动检测编码读取 CSV（兼容 Windows 中文系统的 GBK 编码）"""
    for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            import io
            reader = csv.DictReader(io.StringIO(content))
            return [row for row in reader]
        except (UnicodeDecodeError, UnicodeError):
            continue
    return []


def read_stress_stats(csv_prefix):
    """读取 Locust 压力测试统计 CSV"""
    stats_file = f"{csv_prefix}_stats.csv"
    if not os.path.exists(stats_file):
        return None
    rows = read_csv_auto_encoding(stats_file)
    return rows if rows else None


def read_stress_failures(csv_prefix):
    """读取 Locust 压力测试失败记录"""
    fail_file = f"{csv_prefix}_failures.csv"
    if not os.path.exists(fail_file):
        return []
    return read_csv_auto_encoding(fail_file)


def read_pytest_json(json_path):
    """读取 pytest JSON 报告"""
    if not os.path.exists(json_path):
        return None
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_stress_csv():
    """查找最新的压力测试 CSV 文件"""
    reports_dir = "tests/reports"
    if not os.path.exists(reports_dir):
        return None

    # 查找 stress_ 开头的 stats.csv 文件
    pattern = os.path.join(reports_dir, "stress_*_stats.csv")
    files = glob.glob(pattern)
    if not files:
        return None

    # 按修改时间排序，取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    # 返回去掉 _stats.csv 后缀的前缀
    return files[0].replace("_stats.csv", "")


def generate_html_report(pytest_json=None, stress_csv_prefix=None, output_path=None):
    """生成综合 HTML 测试报告"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if output_path is None:
        reports_dir = "tests/reports"
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

    # ========== 读取功能测试结果 ==========
    pytest_data = read_pytest_json(pytest_json) if pytest_json else None
    func_total = 0
    func_passed = 0
    func_failed = 0
    func_errors = 0
    func_duration = 0
    func_tests = []

    if pytest_data:
        summary = pytest_data.get("summary", {})
        func_total = summary.get("total", 0)
        func_passed = summary.get("passed", 0)
        func_failed = summary.get("failed", 0)
        func_errors = summary.get("error", 0)
        func_duration = pytest_data.get("duration", 0)
        for test in pytest_data.get("tests", []):
            func_tests.append({
                "name": test.get("nodeid", ""),
                "outcome": test.get("outcome", ""),
                "duration": round(test.get("duration", 0), 2),
                "call": test.get("call", {}),
            })

    # ========== 读取压力测试结果 ==========
    stress_stats = None
    stress_failures = []
    if stress_csv_prefix:
        stress_stats = read_stress_stats(stress_csv_prefix)
        stress_failures = read_stress_failures(stress_csv_prefix)

    # ========== 生成 HTML ==========
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能课堂系统 - 测试报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        /* 头部 */
        .header {{
            background: linear-gradient(135deg, #1a73e8, #0d47a1);
            color: white;
            padding: 30px 40px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(26, 115, 232, 0.3);
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; font-size: 14px; }}

        /* 概览卡片 */
        .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .card .number {{ font-size: 36px; font-weight: bold; margin: 8px 0; }}
        .card .label {{ font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        .card.green .number {{ color: #34a853; }}
        .card.red .number {{ color: #ea4335; }}
        .card.blue .number {{ color: #1a73e8; }}
        .card.orange .number {{ color: #f9ab00; }}
        .card.purple .number {{ color: #9334e6; }}

        /* 区块 */
        .section {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }}
        .section h2 {{
            font-size: 20px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f0f2f5;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* 表格 */
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            background: #f8f9fa;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e8eaed;
        }}
        td {{
            padding: 10px 16px;
            border-bottom: 1px solid #f0f2f5;
            font-size: 14px;
        }}
        tr:hover {{ background: #f8f9fa; }}

        /* 状态标签 */
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-pass {{ background: #e6f4ea; color: #137333; }}
        .badge-fail {{ background: #fce8e6; color: #c5221f; }}
        .badge-error {{ background: #fef7e0; color: #b06000; }}

        /* 进度条 */
        .progress-bar {{
            width: 100%;
            height: 24px;
            background: #e8eaed;
            border-radius: 12px;
            overflow: hidden;
            margin: 16px 0;
            display: flex;
        }}
        .progress-pass {{ background: #34a853; transition: width 0.5s; }}
        .progress-fail {{ background: #ea4335; transition: width 0.5s; }}

        /* 响应时间条 */
        .bar-chart {{ margin: 16px 0; }}
        .bar-row {{ display: flex; align-items: center; margin: 8px 0; gap: 10px; }}
        .bar-label {{ width: 180px; font-size: 13px; text-align: right; flex-shrink: 0; }}
        .bar-fill {{
            height: 28px;
            background: linear-gradient(90deg, #1a73e8, #4285f4);
            border-radius: 6px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 12px;
            font-weight: 600;
            min-width: 60px;
            transition: width 0.5s;
        }}

        /* 底部 */
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 13px;
        }}

        /* 筛选按钮 */
        .filter-bar {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
        .filter-btn {{
            padding: 6px 16px;
            border: 1px solid #dadce0;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: #1a73e8;
            color: white;
            border-color: #1a73e8;
        }}

        /* 服务分组 */
        .service-group {{ margin-bottom: 20px; }}
        .service-header {{
            font-size: 15px;
            font-weight: 600;
            padding: 10px 16px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .service-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}
        .dot-5001 {{ background: #34a853; }}
        .dot-5002 {{ background: #1a73e8; }}
        .dot-5003 {{ background: #f9ab00; }}
    </style>
</head>
<body>
<div class="container">

    <!-- 头部 -->
    <div class="header">
        <h1>🎓 智能课堂系统 - 测试报告</h1>
        <p>生成时间：{timestamp} &nbsp;|&nbsp; 基于计算机视觉的智能课堂行为分析系统</p>
    </div>

    <!-- 概览卡片 -->
    <div class="overview">
        <div class="card green">
            <div class="label">功能测试通过</div>
            <div class="number">{func_passed}</div>
        </div>
        <div class="card {'red' if func_failed + func_errors > 0 else 'green'}">
            <div class="label">功能测试失败</div>
            <div class="number">{func_failed + func_errors}</div>
        </div>
        <div class="card blue">
            <div class="label">功能测试总数</div>
            <div class="number">{func_total}</div>
        </div>"""

    # 压力测试卡片
    if stress_stats:
        total_reqs = sum(int(r.get("Request Count", 0)) for r in stress_stats if r.get("Name") != "Aggregated")
        total_failures = sum(int(r.get("Failure Count", 0)) for r in stress_stats if r.get("Name") != "Aggregated")
        agg = [r for r in stress_stats if r.get("Name") == "Aggregated"]
        avg_rt = round(float(agg[0].get("Average Response Time", 0)), 1) if agg else 0
        fail_rate = round(total_failures / total_reqs * 100, 2) if total_reqs > 0 else 0

        html += f"""
        <div class="card orange">
            <div class="label">压力测试请求数</div>
            <div class="number">{total_reqs}</div>
        </div>
        <div class="card purple">
            <div class="label">平均响应时间</div>
            <div class="number">{avg_rt}<small>ms</small></div>
        </div>
        <div class="card {'red' if fail_rate > 1 else 'green'}">
            <div class="label">请求失败率</div>
            <div class="number">{fail_rate}%</div>
        </div>"""

    html += """
    </div>
"""

    # ========== 功能测试详情 ==========
    if func_tests:
        html += """
    <div class="section">
        <h2>📋 功能测试详情</h2>

        <!-- 通过率进度条 -->
        <div style="margin-bottom: 16px;">
            <strong>通过率："""
        pass_rate = round(func_passed / func_total * 100, 1) if func_total > 0 else 0
        html += f"{pass_rate}%</strong>"
        html += f"""
            <div class="progress-bar">
                <div class="progress-pass" style="width: {pass_rate}%"></div>
            </div>
        </div>

        <div class="filter-bar">
            <button class="filter-btn active" onclick="filterTests('all')">全部 ({func_total})</button>
            <button class="filter-btn" onclick="filterTests('passed')">通过 ({func_passed})</button>
            <button class="filter-btn" onclick="filterTests('failed')">失败 ({func_failed})</button>
        </div>

        <table id="func-tests">
            <thead>
                <tr>
                    <th>测试用例</th>
                    <th>状态</th>
                    <th>耗时 (s)</th>
                </tr>
            </thead>
            <tbody>"""

        # 按服务分组
        services = {"user_center": [], "model_inference": [], "face_recognition": []}
        for t in func_tests:
            name = t["name"]
            if "test_user_center" in name:
                services["user_center"].append(t)
            elif "test_model_inference" in name:
                services["model_inference"].append(t)
            elif "test_face_recognition" in name:
                services["face_recognition"].append(t)

        service_labels = {
            "user_center": ("用户中心服务 (5001)", "dot-5001"),
            "model_inference": ("模型推理服务 (5002)", "dot-5002"),
            "face_recognition": ("人脸识别服务 (5003)", "dot-5003"),
        }

        for svc_key, tests in services.items():
            if not tests:
                continue
            label, dot_class = service_labels[svc_key]
            html += f"""
                <tr class="service-group-row">
                    <td colspan="3" class="service-header">
                        <span class="service-dot {dot_class}"></span> {label}
                    </td>
                </tr>"""
            for t in tests:
                outcome = t["outcome"]
                badge_class = "badge-pass" if outcome == "passed" else ("badge-fail" if outcome == "failed" else "badge-error")
                badge_text = "✓ 通过" if outcome == "passed" else ("✗ 失败" if outcome == "failed" else "⚠ 错误")
                short_name = t["name"].split("::")[-1] if "::" in t["name"] else t["name"]

                # 获取错误信息
                error_msg = ""
                if outcome in ["failed", "error"]:
                    call_info = t.get("call", {})
                    longrepr = call_info.get("longrepr", "")
                    if longrepr:
                        # 只取最后一行错误信息
                        error_lines = str(longrepr).strip().split("\n")
                        error_msg = error_lines[-1] if error_lines else ""
                        if len(error_msg) > 80:
                            error_msg = error_msg[:80] + "..."

                html += f"""
                <tr data-outcome="{outcome}">
                    <td>
                        <div style="font-weight:500">{short_name}</div>
                        {"<div style='font-size:12px;color:#c5221f;margin-top:2px'>" + error_msg + "</div>" if error_msg else ""}
                    </td>
                    <td><span class="badge {badge_class}">{badge_text}</span></td>
                    <td>{t['duration']}</td>
                </tr>"""

        html += """
            </tbody>
        </table>
    </div>
"""

    # ========== 压力测试详情 ==========
    if stress_stats:
        # 分离 Aggregated 和各接口数据
        agg_row = None
        interface_rows = []
        for row in stress_stats:
            if row.get("Name") == "Aggregated":
                agg_row = row
            else:
                interface_rows.append(row)

        html += """
    <div class="section">
        <h2>🏋️ 压力测试详情</h2>

        <p style="color:#666;margin-bottom:16px;font-size:14px">
            测试工具：Locust &nbsp;|&nbsp; 模拟多个用户同时访问系统各接口，验证高并发下的系统稳定性
        </p>
"""

        # 接口响应时间柱状图
        if interface_rows:
            max_rt = max(float(r.get("Average Response Time", 0)) for r in interface_rows) or 1

            html += """
        <h3 style="font-size:16px;margin:20px 0 12px">接口平均响应时间</h3>
        <div class="bar-chart">
"""
            for row in sorted(interface_rows, key=lambda x: float(x.get("Average Response Time", 0)), reverse=True):
                name = row.get("Name", "")
                avg = float(row.get("Average Response Time", 0))
                width = max(avg / max_rt * 100, 5) if max_rt > 0 else 5
                html += f"""
            <div class="bar-row">
                <div class="bar-label">{name}</div>
                <div class="bar-fill" style="width:{width:.0f}%">{avg:.1f} ms</div>
            </div>"""

            html += """
        </div>
"""

        # 接口统计表格
        html += """
        <h3 style="font-size:16px;margin:20px 0 12px">接口请求统计</h3>
        <table>
            <thead>
                <tr>
                    <th>接口名称</th>
                    <th>请求方法</th>
                    <th>请求数</th>
                    <th>失败数</th>
                    <th>平均响应 (ms)</th>
                    <th>最小 (ms)</th>
                    <th>最大 (ms)</th>
                    <th>中位数 (ms)</th>
                    <th>RPS</th>
                </tr>
            </thead>
            <tbody>
"""
        for row in interface_rows:
            name = row.get("Name", "")
            method = row.get("Type", "")
            req_count = row.get("Request Count", "0")
            fail_count = row.get("Failure Count", "0")
            avg_rt = round(float(row.get("Average Response Time", 0)), 1)
            min_rt = round(float(row.get("Min Response Time", 0)), 1)
            max_rt_val = round(float(row.get("Max Response Time", 0)), 1)
            med_rt = round(float(row.get("Median Response Time", 0)), 1)
            rps = round(float(row.get("Requests/s", 0)), 2)

            fail_class = ' style="color:#ea4335;font-weight:600"' if int(fail_count) > 0 else ""

            html += f"""
                <tr>
                    <td>{name}</td>
                    <td>{method}</td>
                    <td>{req_count}</td>
                    <td{fail_class}>{fail_count}</td>
                    <td>{avg_rt}</td>
                    <td>{min_rt}</td>
                    <td>{max_rt_val}</td>
                    <td>{med_rt}</td>
                    <td>{rps}</td>
                </tr>"""

        # 汇总行
        if agg_row:
            html += f"""
                <tr style="font-weight:bold;background:#f8f9fa">
                    <td>汇总</td>
                    <td>-</td>
                    <td>{agg_row.get("Request Count", "0")}</td>
                    <td>{agg_row.get("Failure Count", "0")}</td>
                    <td>{round(float(agg_row.get("Average Response Time", 0)), 1)}</td>
                    <td>{round(float(agg_row.get("Min Response Time", 0)), 1)}</td>
                    <td>{round(float(agg_row.get("Max Response Time", 0)), 1)}</td>
                    <td>{round(float(agg_row.get("Median Response Time", 0)), 1)}</td>
                    <td>{round(float(agg_row.get("Requests/s", 0)), 2)}</td>
                </tr>"""

        html += """
            </tbody>
        </table>
    </div>
"""

        # 失败记录
        if stress_failures:
            html += """
    <div class="section">
        <h2>⚠️ 压力测试失败记录（Top 20）</h2>
        <table>
            <thead>
                <tr>
                    <th>错误类型</th>
                    <th>错误信息</th>
                    <th>出现次数</th>
                </tr>
            </thead>
            <tbody>
"""
            for i, f in enumerate(stress_failures[:20]):
                error_type = f.get("Error", "")
                error_msg = f.get("Occurrences", "")
                occurrences = f.get("Occurrences", "0")
                html += f"""
                <tr>
                    <td style="color:#c5221f">{error_type[:60]}</td>
                    <td style="font-size:13px">{f.get("Name", "")}</td>
                    <td>{occurrences}</td>
                </tr>"""

            html += """
            </tbody>
        </table>
    </div>
"""

    # ========== 测试结论 ==========
    html += """
    <div class="section">
        <h2>📊 测试结论</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
            <div>
                <h3 style="font-size:15px;margin-bottom:10px">功能测试</h3>
                <ul style="padding-left:20px;color:#555">
"""
    if func_total > 0:
        html += f"<li>共执行 <strong>{func_total}</strong> 个测试用例</li>"
        html += f"<li>通过 <strong style='color:#34a853'>{func_passed}</strong> 个，"
        html += f"失败 <strong style='color:#ea4335'>{func_failed + func_errors}</strong> 个</li>"
        html += f"<li>通过率 <strong>{pass_rate}%</strong></li>"
        html += f"<li>总耗时 <strong>{func_duration:.1f}s</strong></li>"
    else:
        html += "<li>本次未执行功能测试</li>"

    html += """
                </ul>
            </div>
            <div>
                <h3 style="font-size:15px;margin-bottom:10px">压力测试</h3>
                <ul style="padding-left:20px;color:#555">
"""
    if stress_stats and agg_row:
        html += f"<li>总请求数 <strong>{agg_row.get('Request Count', '0')}</strong></li>"
        html += f"<li>平均响应时间 <strong>{round(float(agg_row.get('Average Response Time', 0)), 1)}ms</strong></li>"
        html += f"<li>吞吐量 <strong>{round(float(agg_row.get('Requests/s', 0)), 2)} req/s</strong></li>"
        fail_rate_val = round(int(agg_row.get("Failure Count", 0)) / max(int(agg_row.get("Request Count", 1)), 1) * 100, 2)
        html += f"<li>失败率 <strong>{fail_rate_val}%</strong></li>"
    else:
        html += "<li>本次未执行压力测试</li>"

    html += """
                </ul>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>智能课堂系统 - 自动化测试报告 | 自动生成于 """ + timestamp + """</p>
    </div>
</div>

<script>
function filterTests(type) {
    const rows = document.querySelectorAll('#func-tests tbody tr[data-outcome]');
    const btns = document.querySelectorAll('.filter-btn');
    btns.forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    rows.forEach(row => {
        if (type === 'all') {
            row.style.display = '';
        } else {
            row.style.display = row.dataset.outcome === type ? '' : 'none';
        }
    });
}
</script>
</body>
</html>"""

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成综合测试报告")
    parser.add_argument("--pytest-json", help="pytest JSON 报告路径")
    parser.add_argument("--stress-csv", help="Locust CSV 文件前缀")
    parser.add_argument("--output", help="输出 HTML 路径")
    args = parser.parse_args()

    stress_csv = args.stress_csv or find_latest_stress_csv()
    output = generate_html_report(
        pytest_json=args.pytest_json,
        stress_csv_prefix=stress_csv,
        output_path=args.output,
    )
    print(f"\n报告已生成: {output}")