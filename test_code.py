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

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params.feat_label_dir
        self._dataset_dir = params.dataset_dir
        self._dataset_combination = '{}_{}'.format(params.dataset, 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params.fs
        self._hop_len_s = params.hop_len_s
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params.label_hop_len_s
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_mel_bins = params.nb_mel_bins
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T

        self._dataset = params.dataset
        self._eps = 1e-8
        self._nb_channels = 4

        # Sound event classes dictionary
        self._unique_classes = params.unique_classes
        self._audio_max_len_samples = params.max_audio_len_s * self._fs  # TODO: Fix the audio synthesis code to always generate 60s of
        # audio. Currently it generates audio till the last active sound event, which is not always 60s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        self._max_feat_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))
        self._max_label_frames = int(np.ceil(self._audio_max_len_samples / float(self._label_hop_len)))

    def load_audio(self, audio_path):
        audio, fs = torchaudio.backend.sox_backend.load(audio_path,
                                                        normalization=True, channels_first=False)
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = torch.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1]) * self._eps
            audio = torch.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        print(nb_bins)
        spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):

            stft_ch = librosa.core.stft(np.array(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            stft_torch = torchaudio.transforms.Spectrogram(n_fft=self._nfft, hop_length=self._hop_len,win_length=self._win_len)(audio_input[:, ch_cnt])
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
        print("stft_ch", stft_ch.shape)
        print("stft_torch", stft_torch.shape)
        return spectra

    def get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
            print("log_mel_spectra",log_mel_spectra.shape)
        mel_feat = mel_feat.reshape((linear_spectra.shape[0], self._nb_mel_bins * linear_spectra.shape[-1]))
        return mel_feat

    def mel_spectrogram_torch(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        mel_feat = np.zeros((self._max_feat_frames, self._nb_mel_bins, _nb_ch))
        for ch_cnt in range(_nb_ch):
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self._fs, n_fft=self._nfft,
                                                            win_length=self._win_len, hop_length=self._hop_len,
                                                            n_mels=self._nb_mel_bins)(audio_input[:, ch_cnt])[:, :self._max_feat_frames].T
            mel_feat[:, :, ch_cnt] = mel_spec
        mel_feat = mel_feat.reshape((self._max_feat_frames, self._nb_mel_bins * _nb_ch))
        return mel_feat


if __name__ == "__main__":
    params = parameter.get_params()
    feature = FeatureClass(params)
    waveform, sample_rate = feature.load_audio('dataset/foa_dev/fold1_room1_mix001_ov1.wav')

    print("waveform", waveform.shape)
    b = feature.spectrogram(waveform)
    c = feature.get_mel_spectrogram(b)
    a = feature.mel_spectrogram_torch(waveform)
    # print(a)
    print(np.load('dataset/feat_label/foa_dev/fold1_room1_mix001_ov1.npy'))
