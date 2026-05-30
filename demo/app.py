import os
import tempfile

import streamlit as st
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

from audio_utils import wav_to_spectrogram
from infer_int8 import predict


# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="语音指令分类 Demo",
    page_icon="🎤"
)

# =========================
# 标题
# =========================
st.title("🎤 实时语音指令分类 Demo")

st.write("点击按钮录音，模型将自动预测语音指令")

# =========================
# 阈值调节
# =========================
threshold = st.slider(
    "置信度阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01
)

st.write(f"当前阈值：{threshold:.2f}")

# =========================
# 麦克风录音
# =========================
audio = mic_recorder(
    start_prompt="开始录音",
    stop_prompt="停止录音",
    just_once=True
)

# =========================
# 推理
# =========================
if audio:

    st.audio(audio["bytes"])

    # =========================
    # 保存浏览器录音
    # =========================
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp_input:

        tmp_input.write(audio["bytes"])

        input_path = tmp_input.name

    # =========================
    # 转标准 PCM WAV
    # =========================
    pcm_path = input_path.replace(".wav", "_pcm.wav")

    sound = AudioSegment.from_file(input_path)

    sound = sound.set_channels(1)

    sound = sound.set_frame_rate(16000)

    sound.export(
        pcm_path,
        format="wav"
    )

    # =========================
    # wav -> spectrogram
    # =========================
    img_path = pcm_path.replace(".wav", ".jpg")

    wav_to_spectrogram(
        pcm_path,
        img_path
    )

    # =========================
    # 显示频谱图
    # =========================
    st.image(
        img_path,
        caption="Mel Spectrogram"
    )

    # =========================
    # 模型推理
    # =========================
    with st.spinner("模型推理中..."):

        label, score = predict(img_path)

    # =========================
    # 阈值判断
    # =========================
    if score >= threshold:

        st.success(
            f"预测结果：{label}"
        )

    else:

        st.warning(
            "置信度过低，请重新录音"
        )

    # =========================
    # confidence
    # =========================
    st.metric(
        label="Confidence",
        value=f"{score:.4f}"
    )

    # =========================
    # 删除临时文件
    # =========================
    try:

        os.remove(input_path)

        os.remove(pcm_path)

        os.remove(img_path)

    except:

        pass