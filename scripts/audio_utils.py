import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def wav_to_spectrogram(wav_path, save_path):

    y, sr = librosa.load(wav_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    plt.figure(figsize=(4,4))

    librosa.display.specshow(
        mel_db,
        sr=sr
    )

    plt.axis("off")

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()