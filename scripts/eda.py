import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
from collections import Counter

DATA_DIR = "xfyun_cold_start/data/processedimage"

# ======================
# 统计类别样本数
# ======================
class_counts = {}
labels = [d for d in os.listdir(DATA_DIR) if d.isdigit()]
labels = sorted(labels, key=lambda x: int(x))
for label in labels:
    label_path = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_path) and label.isdigit():
        class_counts[label] = len(os.listdir(label_path))

# 打印统计信息
print("类别数量:", len(class_counts))
print("各类别样本数:")
for k, v in class_counts.items():
    print(f"Class {k}: {v}")

# ======================
# 画柱状图
# ======================
labels = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(12,6))
plt.bar(labels, counts)

plt.xlabel("Class Label")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")

plt.xticks(rotation=90)
plt.tight_layout()

# 保存图片
plt.savefig("class_distribution.png")