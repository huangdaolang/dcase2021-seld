from torch.utils.data import Dataset
import parameter
import numpy as np
import os
import feature_class
import random
import torch
import torchaudio

# dataset should be 4000 x 7 x 300 x 64 (mel input), label should be 400 x 2 x 60 x 14 (another 42)
class Tau_nigens(Dataset):
    def __init__(self, params, split, shuffle=True, is_eval=False):
        self.params = params
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params.batch_size
        self._feature_seq_len = params.feature_sequence_length
        self._label_seq_len = params.label_sequence_length
        self._shuffle = shuffle
        self._feat_cls = feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._raw_dir = self._feat_cls.get_raw_dir()
        self._input = params.input

        self._filenames_list = list()
        self._filenames_list_raw = list()
        self._nb_frames_file = 0  # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None  # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = self._feat_cls.get_nb_classes()
        self._get_filenames_list_and_feat_label_sizes()

        self._feature_batch_seq_len = self._batch_size * self._feature_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len
        self._len_file = len(self._filenames_list)
        self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._feature_batch_seq_len))))

        if self._input == "mel":
            self.data = self.get_all_data_mel()
        elif self._input == "raw":
            self.data = self.get_all_data_raw()
        self.label = self.get_all_label()

        if self._shuffle:
            random.shuffle(self._filenames_list)

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params.dataset, split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_all_data_mel(self):
        print("start to get feature data")
        data = []
        for file in self._filenames_list:
            temp_feat = np.load(os.path.join(self._feat_dir, file))
            temp = []
            for row in temp_feat:
                temp.append(row)
                if len(temp) == self._feature_seq_len:
                    data.append(temp)
                    temp = []
        data = np.array(data)
        data = np.reshape(data, (int(self._len_file*(self._nb_frames_file/self._feature_seq_len)),
                                 self._feature_seq_len, self._nb_mel_bins, self._nb_ch))
        data = np.transpose(data, (0, 3, 1, 2))
        print("input shape", data.shape)
        return data

    def get_all_data_raw(self):
        print("start to get fetch audio data")
        data = []
        for file in self._filenames_list_raw:
            audio_path = os.path.join(self._raw_dir, file)
            raw, fs = self._feat_cls.load_audio(audio_path)
            segmentation = raw.shape[0] // 10
            for i in range(10):
                data.append(raw[i*segmentation:(i+1)*segmentation].numpy().T)
        data = np.array(data)
        print("input shape", data.shape)
        return torch.tensor(data, dtype=torch.double)

    def get_all_label(self):
        label = []
        for file in self._filenames_list:
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
            label[:, :, :self._nb_classes],  # SED labels
            label  # SED + DOA labels
        ]
        return label

    def __getitem__(self, index):
        entry = {"feature": self.data[index], "label": [self.label[0][index], self.label[1][index]]}
        return entry

    def __len__(self):
        return len(self.data)

    def _get_filenames_list_and_feat_label_sizes(self):
        for filename in os.listdir(self._feat_dir):
            if self._is_eval:
                self._filenames_list.append(filename)
            else:
                if int(filename[4]) in self._splits:  # check which split the file belongs to
                    self._filenames_list.append(filename)

        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))

        self._nb_frames_file = temp_feat.shape[0]
        self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins

        for filename in os.listdir(self._raw_dir):
            if self._is_eval:
                self._filenames_list_raw.append(filename)
            else:
                if int(filename[4]) in self._splits:  # check which split the file belongs to
                    self._filenames_list_raw.append(filename)

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))

            self._label_len = temp_label.shape[-1]
            self._doa_len = (self._label_len - self._nb_classes) // self._nb_classes

        return

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()


if __name__ == "__main__":
    params = parameter.get_params()
    dataset = Tau_nigens(params, split=[3, 4, 5, 6])
