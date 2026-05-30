import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def wav_to_spectrogram(wav_path, save_path):

    # =========================
    # 读取 PCM WAV
    # =========================
    y, sr = sf.read(
        wav_path,
        dtype='float32'
    )

    # 双声道 -> 单声道
    if len(y.shape) > 1:

        y = np.mean(
            y,
            axis=1
        )

    # =========================
    # Mel Spectrogram
    # =========================
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # =========================
    # 保存图片
    # =========================
    plt.figure(figsize=(4, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()