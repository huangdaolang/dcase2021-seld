from collections import namedtuple
import random
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio import transforms
import parameter
import data_loader


def tensor_to_img(spectrogram, transformed_spec):
    plt.figure(figsize=(30, 20))
    plt.subplot(2, 1, 1)
    plt.imshow(spectrogram[0].T)
    plt.subplot(2, 1, 2)
    plt.imshow(transformed_spec[0].T)
    plt.savefig("images/spec.png")


def freq_mask(spectrogram, F=20, replace_with_zero=True):
    cloned = spectrogram.clone()
    num_mel = cloned.shape[-1]

    f = random.randrange(0, F)
    f_zero = random.randrange(0, num_mel - f)
    # avoids randrange error if values are equal and range is empty
    if f_zero == f_zero + f:
        return cloned

    mask_end = f_zero + f

    cloned[:, :, f_zero:mask_end] = 0

    return cloned


def time_mask(spec, T=30, replace_with_zero=True):
    cloned = spec.clone()
    len_spectro = cloned.shape[1]

    t = random.randrange(0, T)
    t_zero = random.randrange(0, len_spectro - t)

    # avoids randrange error if values are equal and range is empty
    if t_zero == t_zero + t:
        return cloned

    mask_end = t_zero + t

    cloned[:, t_zero:mask_end, :] = 0

    return cloned


if __name__ == "__main__":
    params = parameter.get_params(output=False)
    dataset = data_loader.Tau_Nigens(params, split=[3, 4, 5, 6])
    spec = dataset.data[0]
    masked_spec = time_mask(spec)
    tensor_to_img(spec, masked_spec)
