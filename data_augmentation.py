from collections import namedtuple
import random
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio import transforms
import parameter
import data_loader
import raw_feature_class
import os
import numpy as np


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


def get_filenames_list():
    filenames = []
    for filename in os.listdir("../Datasets/SELD2020/foa_dev"):
        if int(filename[4]) in [3, 4, 5, 6]:  # check which split the file belongs to
            filenames.append(filename)

    return filenames


def get_all_raw_data(feat_cls, filenames_list):
    foa_data = []
    mic_data = []
    for file in filenames_list:
        foa_path = os.path.join("../Datasets/SELD2020/foa_dev", file)
        mic_path = os.path.join("../Datasets/SELD2020/mic_dev", file)
        foa, fs = feat_cls.load_audio(foa_path)
        mic, fs = feat_cls.load_audio(mic_path)

        foa_data.append(foa.numpy().T)
        mic_data.append(mic.numpy().T)

    foa_data = np.array(foa_data)
    mic_data = np.array(mic_data)
    print("\tFOA data shape: [n_samples, channel, audio_samples]:{}\n".format(foa_data.shape))
    print("\tMIC data shape: [n_samples, channel, audio_samples]:{}\n".format(mic_data.shape))

    return foa_data, mic_data, filenames_list


def load_output_format_file(output_format_file, azimuth, elevation):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5:  # read polar coordinates format, we ignore the track count
            _output_dict[_frame_ind].append([int(_words[1]), azimuth[0]*float(_words[3])+azimuth[1], elevation*float(_words[4])])
        elif len(_words) == 6:  # read Cartesian coordinates format, we ignore the track count
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict


def audio_channel_swapping(indicator, feat_cls, foa_data, mic_data, filenames_list):
    for i, file in enumerate(filenames_list):
        print("start to process {}".format(file))
        label_file = file.split('.')[0] + ".csv"
        label_file = os.path.join("../Datasets/SELD2020/metadata_dev", label_file)
        save_file = file.split('.')[0]
        foa = foa_data[i]
        mic = mic_data[i]

        if indicator == 0:
            foa_0 = foa
            mic_0 = mic
            desc_file_polar_0 = load_output_format_file(label_file, [1, 0], 1)
            desc_file_0 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_0)
            label_mat_0 = feat_cls.get_labels_for_file(desc_file_0)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_0)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_0)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_0)

        if indicator == 1:
            foa_1 = np.stack((foa[0], -foa[3], -foa[2], foa[1]))
            mic_1 = np.stack((mic[1], mic[3], mic[0], mic[2]))
            desc_file_polar_1 = load_output_format_file(label_file, [1, -90], -1)
            desc_file_1 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_1)
            label_mat_1 = feat_cls.get_labels_for_file(desc_file_1)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_1)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_1)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_1)

        if indicator == 2:
            foa_2 = np.stack((foa[0], -foa[3], foa[2], -foa[1]))
            mic_2 = np.stack((mic[3], mic[1], mic[2], mic[0]))
            desc_file_polar_2 = load_output_format_file(label_file, [-1, -90], 1)
            desc_file_2 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_2)
            label_mat_2 = feat_cls.get_labels_for_file(desc_file_2)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_2)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_2)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_2)

        if indicator == 3:
            foa_3 = np.stack((foa[0], -foa[1], -foa[2], foa[3]))
            mic_3 = np.stack((mic[1], mic[0], mic[3], mic[2]))
            desc_file_polar_3 = load_output_format_file(label_file, [-1, 0], -1)
            desc_file_3 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_3)
            label_mat_3 = feat_cls.get_labels_for_file(desc_file_3)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_3)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_3)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_3)

        if indicator == 4:
            foa_4 = np.stack((foa[0], foa[3], -foa[2], -foa[1]))
            mic_4 = np.stack((mic[2], mic[0], mic[3], mic[1]))
            desc_file_polar_4 = load_output_format_file(label_file, [1, 90], -1)
            desc_file_4 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_4)
            label_mat_4 = feat_cls.get_labels_for_file(desc_file_4)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_4)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_4)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_4)

        if indicator == 5:
            foa_5 = np.stack((foa[0], foa[3], foa[2], foa[1]))
            mic_5 = np.stack((mic[0], mic[2], mic[1], mic[3]))
            desc_file_polar_5 = load_output_format_file(label_file, [-1, 90], 1)
            desc_file_5 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_5)
            label_mat_5 = feat_cls.get_labels_for_file(desc_file_5)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_5)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_5)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_5)

        if indicator == 6:
            foa_6 = np.stack((foa[0], -foa[1], foa[2], -foa[3]))
            mic_6 = np.stack((mic[3], mic[2], mic[1], mic[0]))
            desc_file_polar_6 = load_output_format_file(label_file, [1, 180], 1)
            desc_file_6 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_6)
            label_mat_6 = feat_cls.get_labels_for_file(desc_file_6)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_6)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_6)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_6)

        if indicator == 7:
            foa_7 = np.stack((foa[0], foa[1], -foa[2], -foa[3]))
            mic_7 = np.stack((mic[2], mic[3], mic[0], mic[1]))
            desc_file_polar_7 = load_output_format_file(label_file, [-1, 180], -1)
            desc_file_7 = feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar_7)
            label_mat_7 = feat_cls.get_labels_for_file(desc_file_7)
            np.save("../Datasets/SELD2020/foa_dev_acs/{}_{}.npy".format(save_file, indicator), foa_7)
            np.save("../Datasets/SELD2020/mic_dev_acs/{}_{}.npy".format(save_file, indicator), mic_7)
            np.save("../Datasets/SELD2020/label_dev_acs/{}_{}.npy".format(save_file, indicator), label_mat_7)


if __name__ == "__main__":
    params = parameter.get_params(output=False)
    feat_cls = raw_feature_class.FeatureClass(params, is_eval=False)
    filenames_list = get_filenames_list()
    foa_data, mic_data, filenames_list = get_all_raw_data(feat_cls, filenames_list)

    for indicator in range(8):
        audio_channel_swapping(indicator, feat_cls, foa_data, mic_data, filenames_list)

