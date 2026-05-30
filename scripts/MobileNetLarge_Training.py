# =========================
# 环境 & 依赖
# =========================
import os
os.environ["GLOG_v"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_call_stack_level"] = "2"
os.environ["FLAGS_use_pinned_memory"] = "False"
import sys
sys.stderr = open(os.devnull, "w")

import random
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from paddle.io import DataLoader
from paddle.vision.models import mobilenet_v3_large
from sklearn.metrics import f1_score
from PIL import Image

# =========================
# 随机种子
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
# ⭐ SpecAugment（核心增强）
# =========================
def spec_augment(img, time_mask=10, freq_mask=8):
    img = np.array(img)

    # Time Mask
    t = random.randint(0, time_mask)
    t0 = random.randint(0, img.shape[1] - t)
    img[:, t0:t0+t, :] = 0

    # Frequency Mask
    f = random.randint(0, freq_mask)
    f0 = random.randint(0, img.shape[0] - f)
    img[f0:f0+f, :, :] = 0

    return Image.fromarray(img)

# =========================
# 数据增强
# =========================
train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),

    T.RandomHorizontalFlip(prob=0.5),

    # T.RandomVerticalFlip(prob=0.5),
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
# 读取数据
# =========================
def load_list(txt_path):
    images, labels = [], []
    with open(txt_path) as f:
        for line in f:
            path, label = line.strip().split()
            images.append(os.path.join(data_root, path))
            labels.append(int(label))
    return images, labels

train_imgs, train_labels = load_list(train_list)
val_imgs, val_labels = load_list(val_list)

# =========================
# Dataset
# =========================
class MyDataset(paddle.io.Dataset):

    def __init__(self, imgs, labels, transform=None, use_specaug=False):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.use_specaug = use_specaug   # ⭐记住这个

    def __getitem__(self, idx):

        img = Image.open(self.imgs[idx]).convert("RGB")

        # ⭐ 加增强
        if self.use_specaug:
            if random.random() < 0.5:   # 建议加概率，防止过强
                img = spec_augment(img)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.imgs)

train_dataset = MyDataset(train_imgs, train_labels, train_transform, use_specaug=True)
val_dataset = MyDataset(val_imgs, val_labels, val_transform, use_specaug=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

# =========================
# ⭐ Mixup（优化版）
# =========================
def mixup_data(x, y, alpha=0.1):  # ⭐ 提高强度
    lam = np.random.beta(alpha, alpha)
    batch_size = x.shape[0]
    index = paddle.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =========================
# 模型
# =========================
model = mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Linear(1280, 24)

from paddleslim.analysis import flops

# FLOPs
flops_count = flops(model, [1, 3, 224, 224])
flops_count = float(flops_count)

# 参数量
param_count = sum(int(p.numel()) for p in model.parameters())

# 模型大小
model_size = param_count * 4 / (1024 ** 2)

print("================================")
print("Model:", model.__class__.__name__)
print(f"Params: {param_count/1e6:.2f} M")
print(f"Model Size: {model_size:.2f} MB")
print(f"FLOPs: {flops_count/1e9:.2f} GFLOPs")
print("================================")

# =========================
# 损失函数
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# =========================
# 学习率 & 优化器
# =========================
epochs = 120
history = []

scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=0.06,  # ⭐略微提高
    T_max=epochs
)

optimizer = paddle.optimizer.Momentum(
    learning_rate=scheduler,
    momentum=0.9,
    weight_decay=3e-4,
    parameters=model.parameters()
)

# =========================
# 训练
# =========================

best_f1 = 0
best_Train_Loss = 0
best_Train_acc = 0
best_Val_acc = 0
best_epoch = 0

# Early Stopping
patience = 30        # 允许30个epoch不提升
counter = 0

for epoch in range(epochs):
    print("start training epoch", epoch+1)

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    for step, (images, labels) in enumerate(train_loader):

        if epoch < epochs * 0.8:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            preds = model(images)
            loss = mixup_criterion(criterion, preds, labels_a, labels_b, lam)
        else:
            preds = model(images)
            loss = criterion(preds, labels)

        loss.backward()

        optimizer.step()
        optimizer.clear_grad()

        train_loss += loss.numpy().item()

        pred = preds.argmax(axis=1)

        train_total += labels.shape[0]

        if epoch < epochs * 0.8:
            train_correct += (pred == labels_a).sum().numpy()
        else:
            train_correct += (pred == labels).sum().numpy()

        if step % 10 == 0:
            print(f"Epoch {epoch+1} Step {step} Loss {loss.numpy().item():.4f}")

    train_acc = train_correct / train_total

    # =========================
    # 验证
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with paddle.no_grad():

        for images, labels in val_loader:

            preds = model(images)

            pred = preds.argmax(axis=1)

            val_correct += (pred == labels).sum().numpy()
            val_total += labels.shape[0]

            all_preds.extend(pred.numpy())
            all_labels.extend(labels.numpy())

    val_acc = val_correct / val_total

    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("========================================")
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train_loss / len(train_loader)
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print("LR:", optimizer.get_lr())
    print("========================================")

    history.append({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_acc": val_acc,
    "macro_f1": macro_f1
    })
    
    import pandas as pd
    pd.DataFrame(history).to_csv("/kaggle/working/MobileNetV3Large_training_log.csv", index=False)
    print("Training log saved.")
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_Train_Loss = train_loss
        best_Train_acc = train_acc
        best_Val_acc = val_acc
        best_epoch = epoch + 1

        paddle.save(
        model.state_dict(),
        "/kaggle/working/MobileNetV3Large_best_model.pdparams"
        )
        
        counter = 0   # 有提升，清零
        
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter}/{patience}")
        print("========================================")
    
    scheduler.step()

    # Early Stopping
    if counter >= patience:
        print("========================================")
        print("Early stopping triggered")
        print(f"Best Epoch : {best_epoch}")
        print(f"Best Macro F1 : {best_f1:.4f}")
        print("========================================")
        break

# =========================
# 最优结果
# =========================
print("========================================")
print("Training Finished")
print(f"Best Epoch : {best_epoch}")
print(f"Best Train Loss : {best_Train_Loss:.4f}")
print(f"Best Val Acc : {best_Val_acc:.4f}")
print(f"Best Train Acc : {best_Train_acc:.4f}")
print(f"Best Macro F1 : {best_f1:.4f}")
print("========================================")


# =========================
# Loss 曲线
# =========================
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("/kaggle/working/MobileNetV3small_training_log.csv")
df = pd.read_csv("/kaggle/working/MobileNetV3Large_training_log.csv")

plt.figure(figsize=(8,6))

plt.plot(df["epoch"], df["train_loss"], label="Train Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()
plt.grid(True)

plt.savefig("/kaggle/working/MobileNetV3Large_loss_curve.png", dpi=300)
# plt.savefig("/kaggle/working/MobileNetV3small_loss_curve.png", dpi=300)

plt.close()

# =========================
# Accuracy 曲线
# =========================

plt.figure(figsize=(8,6))

plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
plt.plot(df["epoch"], df["val_acc"], label="Val Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")

plt.legend()
plt.grid(True)

plt.savefig("/kaggle/working/MobileNetV3Large_accuracy_curve.png", dpi=300)
# plt.savefig("/kaggle/working/MobileNetV3small_accuracy_curve.png", dpi=300)
plt.close()

# =========================
# F1 曲线
# =========================

plt.figure(figsize=(8,6))

plt.plot(df["epoch"], df["macro_f1"], label="Macro F1")

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Macro F1 Curve")

plt.legend()
plt.grid(True)

plt.savefig("/kaggle/working/MobileNetV3Large_f1_curve.png", dpi=300)
# plt.savefig("/kaggle/working/MobileNetV3small_f1_curve.png", dpi=300)
plt.close()

print("Training curves saved.")

plt.close()
 