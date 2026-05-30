import os
import random
import shutil
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 配置
# =============================
RANDOM_SEED = 42
FEW_SHOT_NUM = 20

BASE_DIR = "data/HK_Data/zh-HK"
TSV_PATH = os.path.join(BASE_DIR, "validated.tsv")
AUDIO_DIR = os.path.join(BASE_DIR, "clips")
OUTPUT_DIR = os.path.join(BASE_DIR, "processedimage_HK")

random.seed(RANDOM_SEED)

# =============================
# 标签定义（最终版）
# =============================
KEYWORDS = {
    "question": ["？", "點解", "點樣", "點", "幾時"],
    "emotion": ["開心", "嬲", "驚", "傷心", "煩", "慘"],
    "statement": ["係", "真係", "可以", "應該"]
}

LABEL_MAP = {k: i for i, k in enumerate(KEYWORDS.keys())}

# =============================
# 打标签（强过滤）
# =============================
def assign_label(text):
    if pd.isna(text) or len(text) < 5:
        return None

    for label, words in KEYWORDS.items():
        for w in words:
            if w in text:
                return LABEL_MAP[label]

    return None  # ❗ 丢弃无法匹配的（关键）

# =============================
# 清空目录
# =============================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# 读取数据
# =============================
print("📥 读取TSV...")
df = pd.read_csv(TSV_PATH, sep="\t")

# =============================
# 打标签
# =============================
df["label"] = df["sentence"].apply(assign_label)
df = df[df["label"].notnull()].copy()

print("\n📊 标签分布（过滤后）：")
print(df["label"].value_counts())

# =============================
# few-shot划分（严格每类20）
# =============================
train_df, val_df, test_df = [], [], []

labels = sorted(df["label"].unique())

for label in labels:
    subset = df[df["label"] == label].copy()
    subset = subset.sample(frac=1, random_state=RANDOM_SEED)

    if len(subset) < FEW_SHOT_NUM + 10:
        print(f"⚠️ 类别 {label} 样本不足，跳过")
        continue

    train_part = subset.iloc[:FEW_SHOT_NUM]
    val_part = subset.iloc[FEW_SHOT_NUM:FEW_SHOT_NUM + 5]
    test_part = subset.iloc[FEW_SHOT_NUM + 5:FEW_SHOT_NUM + 10]

    train_df.append(train_part)
    val_df.append(val_part)
    test_df.append(test_part)

train_df = pd.concat(train_df)
val_df = pd.concat(val_df)
test_df = pd.concat(test_df)

print("\n📊 Few-shot结果：")
print("Train:", train_df["label"].value_counts())
print("Val  :", val_df["label"].value_counts())

# =============================
# 音频转mel
# =============================
def audio_to_mel(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(2.24, 2.24))
        plt.axis('off')
        plt.imshow(mel_db, aspect='auto', origin='lower')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print("❌ 错误:", audio_path, e)

# =============================
# 生成数据
# =============================
def process(df_split, name):
    records = []
    print(f"\n🎧 生成 {name}...")

    for row in df_split.itertuples():
        audio_path = os.path.join(AUDIO_DIR, row.path)
        if not os.path.exists(audio_path):
            continue

        label = int(row.label)
        save_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(save_dir, exist_ok=True)

        save_name = row.path.split(".")[0] + ".png"
        save_path = os.path.join(save_dir, save_name)

        audio_to_mel(audio_path, save_path)
        records.append(f"{label}/{save_name} {label}")

    print(f"✅ {name} 完成: {len(records)}")
    return records

train_list = process(train_df, "Train")
val_list = process(val_df, "Val")
test_list = process(test_df, "Test")

# =============================
# 保存
# =============================
def write_txt(name, data):
    with open(os.path.join(OUTPUT_DIR, name), "w") as f:
        f.write("\n".join(data))

write_txt("train_list.txt", train_list)
write_txt("val_list.txt", val_list)
write_txt("test_list.txt", test_list)

print("\n🚀 高质量语义few-shot数据生成完成！")