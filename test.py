from models import ReSE_SampleCNN
from torch.utils.data import Dataset
import parameter
import numpy as np
import os
import feature_class
import random
import torch
import utils.utils_functions as utils
from torch.utils.data import DataLoader
from metrics import SELD_evaluation_metrics


class Tau_Nigens_raw_test(Dataset):
    def __init__(self, parameters, shuffle=False, is_eval=True, is_val=False):
        self._params = parameters
        self._is_eval = is_eval
        self._is_val = is_val
        self._label_seq_len = self._params.label_sequence_length
        self._shuffle = shuffle
        self._feat_cls = feature_class.FeatureClass(params=self._params, is_eval=self._is_eval)

        self.foa_dir = "../Datasets/SELD2021/foa_eval"
        self.mic_dir = "../Datasets/SELD2021/mic_eval"
        self.nb_classes = self._feat_cls.get_nb_classes()

        self.filenames_list = list()
        self._nb_frames_file = 0
        self.nb_ch = None
        self.get_filenames_list()  # update above parameters
        self._overlap_size = 6

        # slice length of label and raw audio
        self.data_slice_length = 144000
        # interval value

        self.data_interval = self.data_slice_length
        # whole iteration number
        self.iteration = 10

        self._filter = utils.FilterByOctaves(fs=24000, backend='scipy')

        self.slice_list = self.create_slice_list()
        self.data = self.get_all_data(self.filenames_list)
        print("\tData frames shape: [n_samples, channel, audio_samples]:{}\n".format(self.data.shape))
        self._len_file = len(self.filenames_list)

        print(
            '\tfiles number: {}, classes number:{}\n'
            '\tchannels number: {}, label length per sequence: {}\n'.format(
                self._len_file,  self.nb_classes, self.nb_ch, self._label_seq_len))

    def get_all_data(self, filename_list):
        data = []
        for file in filename_list:
            foa_path = os.path.join(self.foa_dir, file)
            mic_path = os.path.join(self.mic_dir, file)
            foa, fs = self._feat_cls.load_audio(foa_path)  # foa [1440000 x 4]
            mic, fs = self._feat_cls.load_audio(mic_path)  # mic [1440000 x 4]
            foa_mic = np.concatenate([mic.numpy(), foa.numpy()], axis=1)  # foa_mic [1440000 x 8]
            data.append(foa_mic)
        data = np.array(data)
        return torch.tensor(data, dtype=torch.float)

    def __getitem__(self, index):
        data_index = index // self.iteration
        slice_index = index % self.iteration
        data = self.data[data_index, slice_index * self.data_interval: slice_index * self.data_interval + self.data_slice_length, :].T

        entry = {"feature": data}
        return entry

    def __len__(self):
        return len(self.slice_list)

    def get_filenames_list(self):
        self.nb_ch = 8
        for filename in os.listdir(self.foa_dir):
            if "mix" in filename:
                self.filenames_list.append(filename)

    def create_slice_list(self):
        slice_list = list(range(0, len(self.filenames_list) * self.iteration))
        return slice_list

    def get_nb_classes(self):
        return self.nb_classes

    def get_filelist(self):
        return self.filenames_list


params = parameter.get_params()
feature = feature_class.FeatureClass(params)
model = ReSE_SampleCNN.ReSE_SampleCNN(params, ReSE_SampleCNN.Basic_Block)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model.load_state_dict(torch.load("trained_model/06121517_rese.pth", map_location=torch.device('cpu')))
model.eval()

dcase_dump_folder = os.path.join(params.dcase_dir, 'result')
utils.create_folder(dcase_dump_folder)

data_test = Tau_Nigens_raw_test(parameters=params, shuffle=False, is_val=True, is_eval=True)
file_list = data_test.get_filelist()

test_dataloader = DataLoader(data_test, batch_size=10, shuffle=False, drop_last=False)


for i, data in enumerate(test_dataloader):
    feat = data['feature'].to(device)
    test_pred = model(feat)
    # print(test_pred[0, :, 12:24])
    sed_out, doa_out = utils.get_accdoa_labels(test_pred.cpu().detach().numpy(), 12)
    sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(sed_out)
    doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(doa_out)

    print(doa_pred.shape)

    output_file = os.path.join(dcase_dump_folder, file_list[i].replace('.wav', '.csv'))

    output_dict = feature.regression_label_format_to_output_format(
        sed_pred, doa_pred
    )
    output_dict = feature.convert_output_format_cartesian_to_polar(output_dict)
    feature.write_output_format_file(output_file, output_dict)