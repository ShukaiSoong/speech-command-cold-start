!pip install paddlepaddle-gpu -q
!pip install paddleslim -q

import os
os.environ["GLOG_v"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_call_stack_level"] = "2"
os.environ["FLAGS_use_pinned_memory"] = "False"

import sys
sys.stderr = open(os.devnull, "w")

import pandas as pd
import numpy as np
import random
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import DataLoader
from paddle.vision.models import mobilenet_v3_large
from sklearn.metrics import f1_score
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# =========================
# 固定随机种子
# =========================
paddle.seed(42)
np.random.seed(42)
random.seed(42)

device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
paddle.set_device(device)

# =========================
# 路径
# =========================
data_root = "/kaggle/input/datasets/skysonsoong/xunfei-data/data/processedimage"
train_list = os.path.join(data_root, "train_list.txt")
val_list = os.path.join(data_root, "val_list.txt")

# =========================
# 数据增强
# =========================

train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(prob=0.5),
    T.RandomVerticalFlip(prob=0.5),
    T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4
    ),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# 数据加载
# =========================
def load_list(txt_path):
    imgs, labels = [], []
    with open(txt_path) as f:
        for line in f:
            path, label = line.strip().split()
            imgs.append(os.path.join(data_root, path))
            labels.append(int(label))
    return imgs, labels

train_imgs, train_labels = load_list(train_list)
val_imgs, val_labels = load_list(val_list)

# =========================
# Dataset
# =========================

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = paddle.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MyDataset(paddle.io.Dataset):

    def __init__(self, imgs, labels, transform=None):

        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):

        img = Image.open(self.imgs[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.imgs)


train_dataset = MyDataset(train_imgs, train_labels, train_transform)
val_dataset = MyDataset(val_imgs, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# =========================
# 训练函数
# =========================
def train_one_config(LR, WD, num_trials, epochs=30, batch_size=64):

    print(f"\n==== num_trials = {num_trials}, LR={LR}, WD={WD} ====")

    train_loader = DataLoader(
        MyDataset(train_imgs, train_labels, train_transform),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        MyDataset(val_imgs, val_labels, val_transform),
        batch_size=batch_size
    )

    # 模型
    model = mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(1280, 24)

    # 损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=LR, T_max=epochs
    )

    optimizer = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        momentum=0.9,
        weight_decay=WD,
        parameters=model.parameters()
    )

    best_f1 = 0

    for epoch in range(epochs):
        model.train()

        for images, labels in train_loader:
            images, y_a, y_b, lam = mixup_data(images, labels)

            preds = model(images)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with paddle.no_grad():
            for images, labels in val_loader:
                preds = model(images)
                pred = preds.argmax(axis=1)

                all_preds.extend(pred.numpy())
                all_labels.extend(labels.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}, F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1

        scheduler.step()

    return best_f1


# =========================
#随机搜索
# =========================
import random

base_lr = 0.015
base_wd = 5e-4
results = []
num_trials = 12


for i in range(num_trials):
    # log-space 局部扰动
    lr = base_lr * (10 ** random.uniform(-0.15, 0.15))
    wd = base_wd * (10 ** random.uniform(-0.3, 0.3))

    f1 = train_one_config(lr, wd, i+1)

    results.append({
        "LR": lr,
        "WD": wd,
        "F1": f1
    })
# =========================
# 保存结果
# =========================
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/hyperparam_search.csv", index=False)

print("\n===== 最优结果 =====")
print(df.sort_values(by="F1", ascending=False).head())