import os
import cv2
import numpy as np
import paddle

from paddle.io import Dataset, DataLoader
from paddleslim.quant import quant_post_static

paddle.enable_static()

# =========================
# calibration 图片路径
# =========================
img_dir = "calib_images"

img_list = os.listdir(img_dir)

# =========================
# preprocess
# =========================
def preprocess(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    img = img.astype("float32") / 255.0

    mean = np.array([0.485,0.456,0.406], dtype="float32")
    std = np.array([0.229,0.224,0.225], dtype="float32")

    img = (img - mean) / std

    img = img.transpose((2,0,1))

    return img

# =========================
# Dataset
# =========================
class CalibDataset(Dataset):

    def __init__(self, img_dir, img_list):

        self.img_dir = img_dir
        self.img_list = img_list

    def __getitem__(self, idx):

        img_name = self.img_list[idx]

        img_path = os.path.join(self.img_dir, img_name)

        img = preprocess(img_path)

        return {"x": img}

    def __len__(self):

        return len(self.img_list)

# =========================
# DataLoader
# =========================
dataset = CalibDataset(img_dir, img_list)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# =========================
# PTQ
# =========================
quant_post_static(

    executor=paddle.static.Executor(paddle.CPUPlace()),

    model_dir="inference",

    quantize_model_path="inference_int8",

    data_loader=data_loader,

    model_filename="mobilenetv3.pdmodel",

    params_filename="mobilenetv3.pdiparams"
)

print("INT8 Quantization Finished!")