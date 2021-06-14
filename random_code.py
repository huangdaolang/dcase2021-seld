import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
# from sklearn.externals import joblib
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
import math
plot.switch_backend('agg')
import torchaudio
import torch
import parameter
import feature_class
import utils.utils_functions as utils
from conformer import ConformerBlock
import librosa.display
from scipy.io.wavfile import write
import sounddevice as sd
import torch.nn as nn

if __name__ == "__main__":
    params = parameter.get_params()
    feature = feature_class.FeatureClass(params)
    # audio, fs = feature.load_audio('../Datasets/SELD/foa_dev/fold1_room1_mix001_ov1.wav')
    # stft = np.abs(np.squeeze(feature.spectrogram(audio[:, :1])))
    # stft = librosa.amplitude_to_db(stft, ref=np.max)
    # librosa.display.specshow(stft.T, sr=fs, x_axis='s', y_axis='linear')
    # plot.show()
    #
    #
    waveform, sample_rate = feature.load_audio('../Datasets/SELD2021/foa_dev/fold1_room1_mix001.wav')
    # waveform = waveform.T
    print(waveform.shape)
    # sd.play(waveform[:, :], 24000, blocking=True)
    filterbank = utils.FilterByOctaves(fs=sample_rate, backend='scipy')

    seq = nn.Sequential(filterbank)
    seq.eval()
    a = seq(waveform[:, :].T)
    b = 0.5*a + 0.5*waveform.T
    # a = filterbank.forward(waveform[:, 0:1])

    # print(a)
    # print(a[1])
    #
    # print("waveform", waveform.shape)
    # b = feature.spectrogram(waveform)
    # c = feature.get_mel_spectrogram(b)
    # a = feature.mel_spectrogram_torch(waveform)
    # # print(a)
    # print(np.load('dataset/feat_label/foa_dev/fold1_room1_mix001_ov1.npy'))
    #
    # a = np.load("../Datasets/SELD2021/feat_label/foa_dev_label/fold1_room1_mix001.npy")
    # print(a[200, :])

    sd.play(b.T, 24000, blocking=True)
