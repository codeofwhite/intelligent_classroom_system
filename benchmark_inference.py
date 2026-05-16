"""
推理性能基准测试脚本
对比 PyTorch 原始推理 vs OpenVINO 加速推理在 CPU 上的性能差异
输出：FPS、延迟、模型大小、内存占用、CPU利用率
"""
import os
import sys
import time
import json
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================
MODEL_PT_PATH = "runs_6class/yolov8m_balance/weights/best.pt"
OPENVINO_EXPORT_DIR = "runs_6class/yolov8m_balance/weights/best_openvino_model"
IMG_SIZE = 640
NUM_WARMUP = 10        # 预热帧数
NUM_BENCHMARK = 200    # 正式测试帧数
DEVICE = "cpu"

# ============================================================
# 工具函数
# ============================================================
def get_file_size_mb(path):
    """获取文件/目录大小(MB)"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total / (1024 * 1024)
    return 0

def get_model_params(model):
    """获取模型参数量"""
    try:
        info = model.info()
        if isinstance(info, (list, tuple)) and len(info) > 0:
            if isinstance(info, dict):
                return info.get("parameters", 0)
            elif isinstance(info, (list, tuple)):
                return info[0] if isinstance(info[0], (int, float)) else 0
        return 0
    except:
        return 0

def create_dummy_image(size=640):
    """创建一张随机测试图片"""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return img

def measure_memory():
    """获取当前进程内存占用(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_inference(model, num_warmup, num_benchmark, img_size):
    """通用推理基准测试函数"""
    dummy_img = create_dummy_image(img_size)

    # 预热
    for _ in range(num_warmup):
        model.predict(dummy_img, verbose=False, device=DEVICE)

    # 正式测试
    latencies = []
    for _ in range(num_benchmark):
        start = time.perf_counter()
        model.predict(dummy_img, verbose=False, device=DEVICE)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # 转换为 ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
        "num_frames": num_benchmark,
    }

# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  推理性能基准测试 - PyTorch vs OpenVINO (CPU)")
    print("=" * 70)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模型路径: {MODEL_PT_PATH}")
    print(f"  输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  预热帧数: {NUM_WARMUP}")
    print(f"  测试帧数: {NUM_BENCHMARK}")
    print(f"  设备: {DEVICE}")
    print("=" * 70)

    from ultralytics import YOLO

    results = {}

    # ----------------------------------------------------------
    # 1. PyTorch 推理测试
    # ----------------------------------------------------------
    print("\n[1/3] 加载 PyTorch 模型...")
    model_pt = YOLO(MODEL_PT_PATH)

    pt_size_mb = get_file_size_mb(MODEL_PT_PATH)
    pt_params = get_model_params(model_pt)
    print(f"  模型大小: {pt_size_mb:.1f} MB")
    print(f"  参数量: {pt_params:,}")

    mem_before = measure_memory()
    print(f"\n[2/3] PyTorch 推理测试 ({NUM_BENCHMARK} 帧)...")
    pt_result = benchmark_inference(model_pt, NUM_WARMUP, NUM_BENCHMARK, IMG_SIZE)
    mem_after_pt = measure_memory()
    pt_result["model_size_mb"] = pt_size_mb
    pt_result["params"] = pt_params
    pt_result["memory_delta_mb"] = mem_after_pt - mem_before
    results["pytorch"] = pt_result

    print(f"  FPS: {pt_result['fps']:.1f}")
    print(f"  平均延迟: {pt_result['mean_ms']:.1f} ± {pt_result['std_ms']:.1f} ms")
    print(f"  P50/P95/P99: {pt_result['median_ms']:.1f} / {pt_result['p95_ms']:.1f} / {pt_result['p99_ms']:.1f} ms")

    # ----------------------------------------------------------
    # 2. 导出 OpenVINO 模型
    # ----------------------------------------------------------
    print(f"\n[3/3] 导出 OpenVINO 模型...")
    if not os.path.exists(OPENVINO_EXPORT_DIR):
        print(f"  导出中: {OPENVINO_EXPORT_DIR}")
        export_path = model_pt.export(format="openvino", imgsz=IMG_SIZE, half=False)
        print(f"  导出完成: {export_path}")
    else:
        print(f"  OpenVINO 模型已存在，跳过导出")

    # ----------------------------------------------------------
    # 3. OpenVINO 推理测试
    # ----------------------------------------------------------
    print(f"\n[4/3] 加载 OpenVINO 模型...")
    model_ov = YOLO(OPENVINO_EXPORT_DIR, task="detect")

    ov_size_mb = get_file_size_mb(OPENVINO_EXPORT_DIR)
    print(f"  模型大小: {ov_size_mb:.1f} MB")

    mem_before_ov = measure_memory()
    print(f"\n[5/3] OpenVINO 推理测试 ({NUM_BENCHMARK} 帧)...")
    ov_result = benchmark_inference(model_ov, NUM_WARMUP, NUM_BENCHMARK, IMG_SIZE)
    mem_after_ov = measure_memory()
    ov_result["model_size_mb"] = ov_size_mb
    ov_result["memory_delta_mb"] = mem_after_ov - mem_before_ov
    results["openvino"] = ov_result

    print(f"  FPS: {ov_result['fps']:.1f}")
    print(f"  平均延迟: {ov_result['mean_ms']:.1f} ± {ov_result['std_ms']:.1f} ms")
    print(f"  P50/P95/P99: {ov_result['median_ms']:.1f} / {ov_result['p95_ms']:.1f} / {ov_result['p99_ms']:.1f} ms")

    # ----------------------------------------------------------
    # 4. 对比汇总
    # ----------------------------------------------------------
    speedup = pt_result["mean_ms"] / ov_result["mean_ms"]
    fps_gain = ov_result["fps"] / pt_result["fps"]

    results["comparison"] = {
        "speedup": round(speedup, 2),
        "fps_gain": round(fps_gain, 2),
        "latency_reduction_pct": round((1 - ov_result["mean_ms"] / pt_result["mean_ms"]) * 100, 1),
    }

    print("\n" + "=" * 70)
    print("  测试结果汇总")
    print("=" * 70)

    header = f"{'指标':<25} {'PyTorch':>15} {'OpenVINO':>15} {'提升':>12}"
    print(header)
    print("-" * 70)
    print(f"{'平均延迟 (ms)':<25} {pt_result['mean_ms']:>14.1f} {ov_result['mean_ms']:>14.1f} {'-' * 12}")
    print(f"{'P50 延迟 (ms)':<25} {pt_result['median_ms']:>14.1f} {ov_result['median_ms']:>14.1f} {'-' * 12}")
    print(f"{'P95 延迟 (ms)':<25} {pt_result['p95_ms']:>14.1f} {ov_result['p95_ms']:>14.1f} {'-' * 12}")
    print(f"{'FPS':<25} {pt_result['fps']:>14.1f} {ov_result['fps']:>14.1f} {fps_gain:>11.2f}x")
    print(f"{'加速比':<25} {'-' * 15} {'-' * 15} {speedup:>11.2f}x")
    print(f"{'延迟降低 (%)':<25} {'-' * 15} {'-' * 15} {results['comparison']['latency_reduction_pct']:>10.1f}%")
    print(f"{'模型大小 (MB)':<25} {pt_size_mb:>14.1f} {ov_size_mb:>14.1f} {'-' * 12}")
    print(f"{'参数量':<25} {pt_params:>14,} {'N/A':>15} {'-' * 12}")
    print("=" * 70)

    # ----------------------------------------------------------
    # 5. 保存结果
    # ----------------------------------------------------------
    output_path = "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存至: {output_path}")

    # 生成 LaTeX 表格片段
    latex_output = f"""% 自动生成的推理性能对比表格
\\begin{{table}}[H]
\\centering
\\caption{{PyTorch与OpenVINO推理性能对比（CPU, {IMG_SIZE}×{IMG_SIZE}）}}
\\label{{tab:inference_benchmark}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{指标}} & \\textbf{{PyTorch}} & \\textbf{{OpenVINO}} \\\\
\\hline
平均延迟 (ms) & {pt_result['mean_ms']:.1f} $\\pm$ {pt_result['std_ms']:.1f} & {ov_result['mean_ms']:.1f} $\\pm$ {ov_result['std_ms']:.1f} \\\\
P50延迟 (ms) & {pt_result['median_ms']:.1f} & {ov_result['median_ms']:.1f} \\\\
P95延迟 (ms) & {pt_result['p95_ms']:.1f} & {ov_result['p95_ms']:.1f} \\\\
FPS & {pt_result['fps']:.1f} & {ov_result['fps']:.1f} \\\\
模型大小 (MB) & {pt_size_mb:.1f} & {ov_size_mb:.1f} \\\\
加速比 & --- & {speedup:.2f}x \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_path = "benchmark_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
    print(f"  LaTeX 表格已保存至: {latex_path}")


if __name__ == "__main__":
    main()