import matplotlib.pyplot as plt
import parameter
import os
import cls_feature_class
import numpy as np
import librosa
import librosa.display
import feature_class


# librosa
params = parameter.get_params()
aud_dir = os.path.join(params.dataset_dir, 'foa_dev')
filename = '../../Datasets/SELD2021/foa_dev/fold1_room1_mix003.wav'

feat_cls = cls_feature_class.FeatureClass(params)
audio, fs = feat_cls.load_audio(filename)
stft = np.abs(np.squeeze(feat_cls.spectrogram(audio[:, :1])))
stft = librosa.amplitude_to_db(stft, ref=np.max)
spec = feat_cls.spectrogram(audio[:, :1])
mel_spec = feat_cls.get_mel_spectrogram(spec)


# torchaudio
feat_cls_torch = feature_class.FeatureClass(params)
audio, fs = feat_cls_torch.load_audio(filename)
spec_torch = feat_cls_torch.spectrogram(audio[:, :1])
stft_torch = np.abs(np.squeeze(feat_cls.spectrogram(audio[:, :1])))
stft_torch = librosa.amplitude_to_db(stft_torch, ref=np.max)
mel_spec_torch = feat_cls_torch.get_mel_spectrogram(spec_torch)


plt.figure(figsize=(30, 20))
ax1 = plt.subplot(2, 2, 1)
img = librosa.display.specshow(stft.T, sr=fs, x_axis='s', y_axis='linear')
plt.xlim([0, 60])
plt.xticks([])
plt.xlabel('')
plt.title('Spectrogram librosa')
plt.colorbar(img, ax=ax1, format="%+2.f dB")

ax2 = plt.subplot(2, 2, 2)
img = librosa.display.specshow(stft_torch.T, sr=fs, x_axis='s', y_axis='linear')
plt.xlim([0, 60])
plt.xticks([])
plt.xlabel('')
plt.title('Spectrogram torchaudio')
plt.colorbar(img, ax=ax2, format="%+2.f dB")

ax3 = plt.subplot(2, 2, 3)
img = librosa.display.specshow(mel_spec.T, sr=fs, x_axis='s', y_axis='linear')
plt.xlim([0, 60])
plt.xticks([])
plt.xlabel('')
plt.title('Mel Spectrogram librosa')
plt.colorbar(img, ax=ax3, format="%+2.f dB")

ax4 = plt.subplot(2, 2, 4)
img2 = librosa.display.specshow(mel_spec_torch.T, sr=fs, x_axis='s', y_axis='linear')
plt.xlim([0, 60])
plt.xticks([])
plt.xlabel('')
plt.title('Mel Spectrogram torchaudio')
plt.colorbar(img2, ax=ax4, format="%+2.f dB")


plt.savefig("../images/sepc_comparison.png")