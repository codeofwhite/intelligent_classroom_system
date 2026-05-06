"""
测试运行脚本
==========================================
一键运行所有测试，生成测试报告

使用方法：
  python tests/run_tests.py              # 运行全部功能测试
  python tests/run_tests.py --smoke      # 只运行冒烟测试
  python tests/run_tests.py --stress     # 运行压力测试（Locust）
  python tests/run_tests.py --report     # 生成 HTML 报告
"""
import subprocess
import sys
import os
import argparse
import time
from datetime import datetime


def run_command(cmd, cwd=None):
    """执行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"执行: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_functional_tests(html_report=False):
    """运行功能测试"""
    print("\n" + "="*60)
    print("  开始运行功能测试")
    print("="*60)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_user_center.py",
        "tests/test_model_inference.py",
        "tests/test_face_recognition.py",
        "-v",
        "--tb=short",
    ]
    
    if html_report:
        report_dir = "tests/reports"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"test_report_{timestamp}.html")
        cmd.extend(["--html", report_path, "--self-contained-html"])
        print(f"\nHTML 报告将保存到: {report_path}")
    
    return run_command(cmd)


def run_stress_test():
    """运行压力测试"""
    print("\n" + "="*60)
    print("  开始运行压力测试 (Locust)")
    print("="*60)
    print("""
压力测试启动说明：
  1. 请确保所有服务已启动（docker-compose up -d）
  2. 浏览器打开 http://localhost:8089 配置测试参数
  3. Web UI 中 Host 字段留空（各 User 类已内置目标地址）
  4. 建议的测试参数：
     - Number of users: 50
     - Ramp up: 10
    """)
    
    # 不传 --host，让每个 Locust User 类使用自己定义的 host
    # Web UI 中 Host 字段留空即可
    cmd = [
        sys.executable, "-m", "locust",
        "-f", "tests/locustfile.py",
    ]
    return run_command(cmd)


def run_stress_headless(users=50, spawn_rate=10, duration="60s"):
    """
    运行压力测试（无头模式）
    不传 --host，让每个 Locust User 类使用自己定义的 host（5001/5002/5003）
    所有服务同时测试，Locust 会自动分配用户到不同 User 类
    """
    results_dir = "tests/reports"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_prefix = os.path.join(results_dir, f"stress_{timestamp}")

    print(f"\n{'='*60}")
    print(f"  压力测试（同时测试三个服务）")
    print(f"  并发用户: {users}, 启动速率: {spawn_rate}/秒, 持续时间: {duration}")
    print(f"  UserCenterUser     → localhost:5001")
    print(f"  ModelInferenceUser → localhost:5002")
    print(f"  FaceRecognitionUser → localhost:5003")
    print(f"  ClassroomScenario  → localhost:5001")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "locust",
        "-f", "tests/locustfile.py",
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "--run-time", duration,
        "--csv", csv_prefix,
    ]
    return run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="智能课堂系统 - 测试运行器")
    parser.add_argument("--smoke", action="store_true", help="只运行冒烟测试")
    parser.add_argument("--stress", action="store_true", help="运行压力测试（Locust Web UI）")
    parser.add_argument("--stress-headless", action="store_true", help="运行压力测试（无头模式）")
    parser.add_argument("--users", type=int, default=50, help="压力测试并发用户数（默认50）")
    parser.add_argument("--duration", type=str, default="60s", help="压力测试持续时间（默认60s）")
    parser.add_argument("--report", action="store_true", help="生成 HTML 测试报告")
    parser.add_argument("--all", action="store_true", help="运行所有测试（功能+压力）")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print(f"\n{'#'*60}")
    print(f"  智能课堂系统 - 自动化测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    if args.stress:
        return_code = run_stress_test()
    elif args.stress_headless:
        return_code = run_stress_headless(args.users, duration=args.duration)
    elif args.all:
        return_code = run_functional_tests(args.report)
        if return_code == 0:
            print("\n功能测试通过，继续运行压力测试...")
            return_code = run_stress_headless(args.users, duration=args.duration)
    else:
        return_code = run_functional_tests(args.report)
    
    print(f"\n{'#'*60}")
    if return_code == 0:
        print("  测试完成！所有测试通过 ✓")
    else:
        print(f"  测试完成！存在失败的测试（退出码: {return_code}）")
    print(f"{'#'*60}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())