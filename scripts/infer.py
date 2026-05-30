import time
import numpy as np
from PIL import Image

import paddle
import paddle.nn.functional as F

# =========================
# 加载模型
# =========================
model = paddle.jit.load("inference/mobilenetv3")

model.eval()

# =========================
# 类别
# =========================
label_map = {
    0: "class0",
    1: "class1",
    2: "class2",
    # ...
}

# =========================
# preprocess
# =========================
def preprocess(img_path):

    img = Image.open(img_path).convert("RGB")

    img = img.resize((256, 256))

    left = (256 - 224) // 2
    top = (256 - 224) // 2

    img = img.crop((left, top, left+224, top+224))

    img = np.array(img).astype("float32") / 255.0

    mean = np.array([0.485,0.456,0.406], dtype="float32")
    std = np.array([0.229,0.224,0.225], dtype="float32")

    img = (img - mean) / std

    img = img.transpose((2,0,1))

    img = np.expand_dims(img, axis=0)

    return paddle.to_tensor(img, dtype='float32')

# =========================
# 推理
# =========================
img = preprocess("/Users/minchen/Desktop/SPEECH Projects/KDXF/xfyun_cold_start/scripts/train_51.jpg")

# warmup
for _ in range(10):
    _ = model(img)

# 正式测速
times = []

for _ in range(100):

    start = time.time()

    output = model(img)

    end = time.time()

    times.append((end-start)*1000)

avg_time = np.mean(times)

# prediction
prob = F.softmax(output, axis=1)

pred = paddle.argmax(prob, axis=1).numpy()[0]

print("Prediction:", label_map[pred])

print(f"Average inference time: {avg_time:.2f} ms")