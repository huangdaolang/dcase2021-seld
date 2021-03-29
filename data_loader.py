from torch.utils.data import Dataset
import parameter
import numpy as np
import os
import feature_class
import random
import torch

''' data output format: n_samples x channels x sequence length x n_mel (4000 x 7 x 300 x 64)
    label output format: list[n_samples x sequence length x 14, n_samples x sequence length x 42]
                             [SED, DOA] 4000 x 60 x (42 and 14)'''


class Tau_Nigens(Dataset):
    def __init__(self, params, split, shuffle=True, is_eval=False):
        self.params = params
        self._input = params.input
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._feature_seq_len = params.feature_sequence_length
        self._label_seq_len = params.label_sequence_length
        self._shuffle = shuffle
        self._feat_cls = feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._raw_dir = self._feat_cls.get_raw_dir()

        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_classes = self._feat_cls.get_nb_classes()

        self._filenames_list = list()
        self._nb_frames_file = 0
        self._nb_ch = None
        self._get_filenames_list_and_feat_label_sizes()  # update above parameters

        if self._shuffle:
            random.shuffle(self._filenames_list)

        self._len_file = len(self._filenames_list)

        # get data
        if self._input == "mel":
            self.data = self.get_all_data_mel()
        elif self._input == "raw":
            self.data = self.get_all_data_raw()
        self.label = self.get_all_label()

        print(
            '\tfiles number: {}, classes number:{}\n'
            '\tnumber of frames per file: {}, mel bins length: {}, channels number: {}\n'
            '\tfeat length per sequence: {}, label length per sequence: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch,
                self._feature_seq_len, self._label_seq_len,
                self._label_dir,
                self._feat_dir))

    def get_all_data_mel(self):
        print("start to fetch mel spectrogram features data")
        data = []
        for file in self._filenames_list:
            features_per_file = np.load(os.path.join(self._feat_dir, file))  # 3000*448 (448 is mel * channel)
            features_per_data = []  # include on piece of data
            for row in features_per_file:
                features_per_data.append(row)
                if len(features_per_data) == self._feature_seq_len:
                    data.append(features_per_data)
                    features_per_data = []
        data = np.array(data)
        data = np.reshape(data, (int(self._len_file*(self._nb_frames_file/self._feature_seq_len)),
                                 self._feature_seq_len, self._nb_mel_bins, self._nb_ch))
        data = np.transpose(data, (0, 3, 1, 2))  # samples, channel, width, height
        print("Data frames shape: [n_samples, channel, width, height]", data.shape)
        return torch.tensor(data, dtype=torch.double)

    def get_all_data_raw(self):
        print("start to get fetch raw audio data")
        data = []
        for file in self._filenames_list:
            audio_path = os.path.join(self._raw_dir, file)
            raw, fs = self._feat_cls.load_audio(audio_path)
            segmentation = raw.shape[0] // 10
            for i in range(10):
                data_seg = raw[i*segmentation:(i+1)*segmentation].numpy().T
                data.append(data_seg)
        data = np.array(data)
        print("Data frames shape: [n_samples, channel, audio_samples]", data.shape)
        return torch.tensor(data, dtype=torch.double)

    def get_all_label(self):
        label = []
        for file in self._filenames_list:
            if self._input == "raw":
                file = file.split('.')[0] + ".npy"
            temp_label = np.load(os.path.join(self._label_dir, file))
            temp = []
            for row in temp_label:
                temp.append(row)
                if len(temp) == self._label_seq_len:
                    label.append(temp)
                    temp = []
        label = np.array(label)
        print("Label shape", label.shape)
        label = [
            torch.tensor(label[:, :, :self._nb_classes]),  # SED labels
            torch.tensor(label[:, :, self._nb_classes:])  # DOA labels
        ]

        return label

    def __getitem__(self, index):
        entry = {"feature": self.data[index], "label": [self.label[0][index], self.label[1][index]]}
        return entry

    def __len__(self):
        return len(self.data)

    def _get_filenames_list_and_feat_label_sizes(self):
        if self._input == "mel":
            for filename in os.listdir(self._feat_dir):
                if self._is_eval:
                    self._filenames_list.append(filename)
                else:
                    if int(filename[4]) in self._splits:  # check which split the file belongs to
                        self._filenames_list.append(filename)
            temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
            self._nb_frames_file = temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins

        elif self._input == "raw":
            for filename in os.listdir(self._raw_dir):
                if self._is_eval:
                    self._filenames_list.append(filename)
                else:
                    if int(filename[4]) in self._splits:  # check which split the file belongs to
                        self._filenames_list.append(filename)

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_filelist(self):
        return self._filenames_list


if __name__ == "__main__":
    params = parameter.get_params()
    dataset = Tau_Nigens(params, split=[3, 4, 5, 6])
