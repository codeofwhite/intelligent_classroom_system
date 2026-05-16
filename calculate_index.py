# 你的数据集每类数量 (按 0~5 顺序)
counts = [13141, 26246, 9984, 3476, 4314, 7518]

# 倒数归一化公式计算类别权重
inv = [1.0 / c for c in counts]
sum_inv = sum(inv)
weights = [ (x / sum_inv) * len(counts) for x in inv ]

# 输出结果
print("精确权重：")
for i, w in enumerate(weights):
    print(f"类别 {i} : {w:.3f}")

# 四舍五入方便训练使用
print("\n最终可直接使用的权重（保留1位小数）：")
print([round(w, 1) for w in weights])