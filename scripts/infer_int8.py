import time
import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor

# =========================
# 创建 predictor
# =========================
config = Config(
    "inference_int8/model.pdmodel",
    "inference_int8/model.pdiparams"
)

config.disable_gpu()

predictor = create_predictor(config)

# =========================
# preprocess
# =========================
def preprocess(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))

    img = img.astype("float32") / 255.0

    mean = np.array([0.485,0.456,0.406], dtype="float32")
    std = np.array([0.229,0.224,0.225], dtype="float32")

    img = (img - mean) / std

    img = img.transpose((2,0,1))

    img = np.expand_dims(img, axis=0)

    return img

# =========================
# 输入图片
# =========================
img = preprocess(
    "/Users/minchen/Desktop/SPEECH Projects/KDXF/xfyun_cold_start/scripts/train_51.jpg"
)

# =========================
# 输入
# =========================
input_handle = predictor.get_input_handle(
    predictor.get_input_names()[0]
)

input_handle.copy_from_cpu(img)

# warmup
for _ in range(10):
    predictor.run()

# =========================
# 测速
# =========================
times = []

for _ in range(100):

    start = time.time()

    predictor.run()

    end = time.time()

    times.append((end-start)*1000)

avg_time = np.mean(times)

# =========================
# 输出
# =========================
output_handle = predictor.get_output_handle(
    predictor.get_output_names()[0]
)

output = output_handle.copy_to_cpu()

pred = np.argmax(output)

print("Prediction:", pred)

print(f"Average inference time: {avg_time:.2f} ms")