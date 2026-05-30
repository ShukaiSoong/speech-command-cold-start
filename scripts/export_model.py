import paddle
import paddle.nn as nn

from paddle.vision.models import mobilenet_v3_large

# =========================
# 创建模型
# =========================
model = mobilenet_v3_large(pretrained=False)

model.classifier[3] = nn.Linear(1280, 24)

# =========================
# 加载参数
# =========================
state_dict = paddle.load(
    "/Users/minchen/Desktop/SPEECH Projects/KDXF/xfyun_cold_start/scripts/MobileNetV3Large_best_model.pdparams"
)

model.set_state_dict(state_dict)

model.eval()

# =========================
# 转静态图
# =========================
model = paddle.jit.to_static(
    model,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None, 3, 224, 224],
            dtype='float32'
        )
    ]
)

# =========================
# 导出 inference model
# =========================
paddle.jit.save(
    model,
    "inference/mobilenetv3"
)

print("Model exported successfully!")