import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor

# =========================
# predictor
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

    img = cv2.resize(img, (224, 224))

    img = img.astype("float32") / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype="float32"
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype="float32"
    )

    img = (img - mean) / std

    img = img.transpose((2, 0, 1))

    img = np.expand_dims(img, axis=0)

    return img

# =========================
# softmax
# =========================
def softmax(x):

    exp_x = np.exp(x - np.max(x))

    return exp_x / exp_x.sum()

# =========================
# predict
# =========================
def predict(img_path):

    img = preprocess(img_path)

    input_handle = predictor.get_input_handle(
        predictor.get_input_names()[0]
    )

    input_handle.copy_from_cpu(img)

    predictor.run()

    output_handle = predictor.get_output_handle(
        predictor.get_output_names()[0]
    )

    output = output_handle.copy_to_cpu()

    probs = softmax(output[0])

    pred = int(np.argmax(probs))

    score = float(probs[pred])

    return pred, score