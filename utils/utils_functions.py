import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import scipy
import scipy.signal


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5
    return sed, accdoa_in


def collect_test_labels(_data_gen_test, _data_out, _nb_classes, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        if _data_gen_test.get_data_gen_mode():
            doa_label = tmp_label[1]
        else:
            doa_label = tmp_label[1][:, :, _nb_classes:]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = doa_label
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FilterByOctaves(nn.Module):
    '''Generates an octave wide filterbank and filters tensors.
    This is gpu compatible if using torch backend, but it is super slow and should not be used at all.

    The octave filterbanks is created using cascade Buttwerworth filters, which then are processed using
    the biquad function native to PyTorch.

    This is useful to get the decay curves of RIRs.'''
    def __init__(self, center_freqs=[250, 500, 1000, 2000, 3000, 4000], order=3, fs=24000, backend='scipy'):
        super(FilterByOctaves, self).__init__()

        self.center_freqs = center_freqs
        self.order = order
        self.fs = fs
        self.backend = backend
        self.sos = []
        for freq in self.center_freqs:
            tmp_sos = self._get_octave_filter(freq, self.fs, self.order)
            self.sos.append(tmp_sos[0])
            self.sos.append(tmp_sos[1])
            self.sos.append(tmp_sos[2])

    def forward_torch(self, x):
        out = []
        # for ii, this_sos in enumerate(self.sos):
        #     tmp = torch.clone(x)
        #     for jj in range(this_sos.shape[0]):
        #         tmp = torchaudio.functional.biquad(tmp,
        #                                            b0=this_sos[jj,0], b1=this_sos[jj,1], b2=this_sos[jj,2],
        #                                            a0=this_sos[jj,3], a1=this_sos[jj,4], a2=this_sos[jj,5])
        #     out.append(torch.clone(tmp))
        # out = torch.stack(out, dim=-2)  # Stack over frequency bands
        this_sos = self.sos[0]
        tmp = x
        jj = 0
        tmp = torchaudio.functional.biquad(tmp,b0=this_sos[jj,0], b1=this_sos[jj,1], b2=this_sos[jj,2],
                                                   a0=this_sos[jj,3], a1=this_sos[jj,4], a2=this_sos[jj,5])

        # tmp = torchaudio.functional.lowpass_biquad(x, 24000, 500, 0.707)
        # out = tmp / tmp.max(dim=-1)[0]
        return tmp

    def forward_scipy(self, x):
        # out = []
        # for ii, this_sos in enumerate(self.sos):
        #     tmp = torch.clone(x).numpy()
        #     tmp = scipy.signal.sosfilt(this_sos, tmp, axis=-1)
        #     out.append(torch.from_numpy(tmp))
        # out = torch.stack(out, dim=-2)  # Stack over frequency bands
        index = np.random.randint(0, len(self.sos))
        this_sos = self.sos[index]

        tmp = torch.clone(x).numpy()
        out = scipy.signal.sosfilt(this_sos, tmp, axis=-1)

        out = out / np.abs(out).max(axis=-1).reshape(-1, 1)
        data = 0.9 * tmp + 0.1 * out
        data = data / np.abs(data).max(axis=-1).reshape(-1, 1)
        data = torch.from_numpy(data)
        return data

    def forward(self, x):
        if self.backend == 'scipy':
            out = self.forward_scipy(x)
        else:
            out = self.forward_torch(x)
        return out

    def get_filterbank_impulse_response(self):
        '''Returns the impulse response of the filterbank.'''
        impulse = torch.zeros(1, self.fs * 20)
        impulse[0, self.fs] = 1
        response = self.forward(impulse)
        return response

    @staticmethod
    def _get_octave_filter(center_freq: float, fs: int, order: int = 3):
        '''
        Design octave band filters with butterworth.
        Returns a sos matrix (tensor) of the shape [filters, 6], in standard sos format.

        Based on octdsgn(Fc,Fs,N); in MATLAB.
        References:
            [1] ANSI S1.1-1986 (ASA 65-1986): Specifications for
                Octave-Band and Fractional-Octave-Band Analog and
                Digital Filters, 1993.
        '''
        beta = np.pi / 2 / order / np.sin(np.pi / 2 / order)
        alpha = (1 + np.sqrt(1 + 8 * beta ** 2)) / 4 / beta
        W1 = center_freq / (fs / 2) * np.sqrt(1 / 2) / alpha
        W2 = center_freq / (fs / 2) * np.sqrt(2) * alpha
        Wn = np.array([W1, W2])

        sos = scipy.signal.butter(N=order, Wn=Wn, btype='bandpass', analog=False, output='sos')
        sos_low = scipy.signal.butter(N=order, Wn=W2, btype='lowpass', analog=False, output='sos')
        sos_high = scipy.signal.butter(N=order, Wn=W1, btype='highpass', analog=False, output='sos')
        return [torch.from_numpy(sos_high), torch.from_numpy(sos_low), torch.from_numpy(sos)]