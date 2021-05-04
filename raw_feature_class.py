import parameter
import os
import numpy as np
import matplotlib.pyplot as plot
import torchaudio
import torch
import utils.utils_functions as utils

plot.switch_backend('agg')
torchaudio.set_audio_backend('sox_io')


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

        # Local parameters
        self._is_eval = is_eval

        self._fs = params.fs

        self._label_hop_len_s = params.label_hop_len_s  # 0.1s
        self._label_hop_len = int(self._fs * self._label_hop_len_s)  # 2400
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._dataset = params.dataset
        self._eps = 1e-8
        self._nb_channels = 4

        # Sound event classes dictionary
        self._unique_classes = params.unique_classes
        self._audio_max_len_samples = params.max_audio_len_s * self._fs

        self._max_label_frames = int(np.ceil(self._audio_max_len_samples / float(self._label_hop_len)))  # 600

    # load the raw audio file
    def load_audio(self, audio_path):
        audio, fs = torchaudio.backend.sox_backend.load(audio_path, normalization=True, channels_first=False)
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = torch.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1]) * self._eps
            audio = torch.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: labels of the format [sed_label, doa_label],
        where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
        where doa_labels is of dimension [nb_frames, 3*nb_classes], nb_classes each for x, y, z axis,
        """

        se_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        x_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        y_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        z_label = np.zeros((self._max_label_frames, len(self._unique_classes)))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[1]
                    y_label[frame_ind, active_event[0]] = active_event[2]
                    z_label[frame_ind, active_event[0]] = active_event[3]

        label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
        return label_mat

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        utils.create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file_polar = self.load_output_format_file(os.path.join(self._desc_dir, file_name))
            desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
            label_mat = self.get_labels_for_file(desc_file)
            print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
            np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 5:  # read polar coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4])])
            elif len(_words) == 6:  # read Cartesian coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                _fid.write(
                    '{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]),
                                                 float(_value[3])))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = len(self._unique_classes)
        _is_polar = _doa_labels.shape[-1] == 2 * _nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2 * _nb_classes]
            _z = _doa_labels[:, 2 * _nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append(
                            [_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append(
                            [_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class],
                             _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[2] * np.pi / 180.
                    azi_rad = tmp_val[1] * np.pi / 180

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], x, y, z])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[1], tmp_val[2], tmp_val[3]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
                    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    out_dict[frame_cnt].append([tmp_val[0], azimuth, elevation])
        return out_dict

    def get_classes(self):
        return self._unique_classes

    def get_raw_dir(self):
        return os.path.join(
            self._dataset_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir, '{}_label'.format(self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return len(self._unique_classes)

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_nb_frames(self):
        return self._max_label_frames


if __name__ == "__main__":
    params = parameter.get_params()
    feature_class = FeatureClass(params)
    print(feature_class)