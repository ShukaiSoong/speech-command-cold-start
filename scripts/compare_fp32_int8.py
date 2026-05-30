import os
import time
import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor

# ======================================
# 数据集根目录
# ======================================
DATA_ROOT = "/Users/minchen/Desktop/SPEECH Projects/KDXF/xfyun_cold_start/data/processedimage"

# ======================================
# preprocess
# ======================================
def preprocess(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, (224,224))

    img = cv2.resize(img, (256,256))

    start = (256 - 224) // 2

    img = img[
        start:start+224,
        start:start+224
    ]

    img = img.astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    img = (img - mean) / std

    img = img.transpose((2, 0, 1))

    img = np.expand_dims(img, axis=0)

    return img


# ======================================
# 创建 predictor
# ======================================
def create_fp32_predictor():

    config = Config(
        "inference/mobilenetv3.pdmodel",
        "inference/mobilenetv3.pdiparams"
    )

    config.disable_gpu()

    predictor = create_predictor(config)

    return predictor


def create_int8_predictor():

    config = Config(
        "inference_int8/model.pdmodel",
        "inference_int8/model.pdiparams"
    )

    config.disable_gpu()

    predictor = create_predictor(config)

    return predictor


# ======================================
# 单模型评测
# ======================================
def evaluate_model(predictor, model_name):

    total = 0
    correct = 0

    latency_list = []

    input_handle = predictor.get_input_handle(
        predictor.get_input_names()[0]
    )

    output_handle = predictor.get_output_handle(
        predictor.get_output_names()[0]
    )

    # 遍历所有类别
    class_dirs = sorted(os.listdir(DATA_ROOT))

    for class_name in class_dirs:

        class_path = os.path.join(DATA_ROOT, class_name)

        if not os.path.isdir(class_path):
            continue

        label = int(class_name)

        image_list = os.listdir(class_path)

        # 每类最多测20张（避免太慢）
        image_list = image_list[:20]

        for image_name in image_list:

            image_path = os.path.join(
                class_path,
                image_name
            )

            try:

                img = preprocess(image_path)

                input_handle.copy_from_cpu(img)

                # warmup
                predictor.run()

                # 正式测速
                start = time.time()

                predictor.run()

                end = time.time()

                latency = (end - start) * 1000

                latency_list.append(latency)

                output = output_handle.copy_to_cpu()

                pred = np.argmax(output)

                total += 1

                if pred == label:
                    correct += 1

            except Exception as e:

                print(f"ERROR: {image_path}")
                print(e)

    accuracy = correct / total

    avg_latency = np.mean(latency_list)

    print("\n====================================")
    print(f"{model_name} RESULT")
    print("====================================")

    print(f"Total Samples: {total}")

    print(f"Correct: {correct}")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Average Latency: {avg_latency:.2f} ms")

    return accuracy, avg_latency


# ======================================
# main
# ======================================
if __name__ == "__main__":

    print("\nCreating FP32 Predictor...")
    fp32_predictor = create_fp32_predictor()

    print("\nCreating INT8 Predictor...")
    int8_predictor = create_int8_predictor()

    # FP32
    fp32_acc, fp32_latency = evaluate_model(
        fp32_predictor,
        "FP32"
    )

    # INT8
    int8_acc, int8_latency = evaluate_model(
        int8_predictor,
        "INT8"
    )

    # ======================================
    # 最终对比
    # ======================================
    print("\n\n====================================")
    print("FINAL COMPARISON")
    print("====================================")

    print(f"FP32 Accuracy: {fp32_acc:.4f}")
    print(f"INT8 Accuracy: {int8_acc:.4f}")

    print()

    print(f"FP32 Latency: {fp32_latency:.2f} ms")
    print(f"INT8 Latency: {int8_latency:.2f} ms")

    print()

    print(
        f"Latency Speedup: "
        f"{fp32_latency / int8_latency:.2f}x"
    )