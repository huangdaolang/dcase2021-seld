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


def freq_mask(spectrogram, F=20, num_masks=1, replace_with_zero=True):
    cloned = spectrogram.clone()
    num_mel = cloned.shape[-1]
    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel - f)
        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            cloned[:, :, f_zero:mask_end] = 0
        else:
            cloned[:, :, f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=30, num_masks=1, replace_with_zero=True):
    cloned = spec.clone()
    len_spectro = cloned.shape[1]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            cloned[:, t_zero:mask_end, :] = 0
        else:
            cloned[:, t_zero:mask_end, :] = cloned.mean()
    return cloned


if __name__ == "__main__":
    params = parameter.get_params(output=False)
    dataset = data_loader.Tau_Nigens(params, split=[3, 4, 5, 6])
    spec = dataset.data[0]
    masked_spec = time_mask(spec)
    tensor_to_img(spec, masked_spec)
