#conda activate xfyun_env
import os
import shutil
import pandas as pd
import random


# ======================
# 配置
# ======================
RANDOM_SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.1

BASE_DIR = "xfyun_cold_start/data"
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "train")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "processedimage")

# ======================
# 固定随机种子
# ======================
random.seed(RANDOM_SEED)

# ======================
# 读取 CSV
# ======================
print("📥 读取 train.csv...")
df = pd.read_csv(CSV_PATH)

print("样本总数:", len(df))
print("类别数:", df["label"].nunique())

# ======================
# 创建类别文件夹
# ======================
print("📁 创建类别文件夹...")
os.makedirs(PROCESSED_DIR, exist_ok=True)

labels = sorted(df["label"].unique())

for label in labels:
    os.makedirs(os.path.join(PROCESSED_DIR, str(label)), exist_ok=True)

print("✅ 类别文件夹创建完成")

# ======================
# 拷贝图片到对应类别
# ======================
print("📦 开始拷贝图片...")

for idx, row in df.iterrows():
    img_name = row["image"]
    label = str(row["label"])

    src_path = os.path.join(RAW_TRAIN_DIR, img_name)
    dst_path = os.path.join(PROCESSED_DIR, label, img_name)

    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

    if idx % 500 == 0:
        print(f"已处理 {idx} 张...")

print("🎉 图片拷贝完成")

# ======================
# 分层划分
# ======================
print("✂ 开始划分数据集...")

train_list = []
val_list = []
test_list = []

for label in labels:
    label_path = os.path.join(PROCESSED_DIR, str(label))
    images = os.listdir(label_path)

    random.shuffle(images)

    total = len(images)

    test_num = round(total * TEST_RATIO)
    val_num = round(total * VAL_RATIO)
    train_num = total - test_num - val_num  # 保证总数精确

    test_imgs = images[:test_num]
    val_imgs = images[test_num:test_num + val_num]
    train_imgs = images[test_num + val_num:]

    for img in train_imgs:
        train_list.append(f"{label}/{img} {label}")

    for img in val_imgs:
        val_list.append(f"{label}/{img} {label}")

    for img in test_imgs:
        test_list.append(f"{label}/{img} {label}")

print("✅ 数据集划分完成")

# ======================
# 写入 txt 文件
# ======================
def write_txt(filename, data):
    with open(os.path.join(PROCESSED_DIR, filename), "w") as f:
        f.write("\n".join(data))

write_txt("train_list.txt", train_list)
write_txt("val_list.txt", val_list)
write_txt("test_list.txt", test_list)

with open(os.path.join(PROCESSED_DIR, "labels.txt"), "w") as f:
    for label in labels:
        f.write(f"{label}\n")

# ======================
# 输出统计信息
# ======================
print("\n📊 最终统计：")
print("Train:", len(train_list))
print("Val:", len(val_list))
print("Test:", len(test_list))
print("Total:", len(train_list) + len(val_list) + len(test_list))

print("\n📁 目录应包含：")
print("labels.txt")
print("train_list.txt")
print("val_list.txt")
print("test_list.txt")

print("\n🚀 数据准备完成！")