import os
import numpy as np
import librosa.display
import feature_class
import parameter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plot

plot.switch_backend('agg')
plot.rcParams.update({'font.size': 22})


def collect_classwise_data(_in_dict):
    _out_dict = {}
    for _key in _in_dict.keys():
        for _seld in _in_dict[_key]:
            if _seld[0] not in _out_dict:
                _out_dict[_seld[0]] = []
            _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2]])
    return _out_dict


def plot_func(plot_data, hop_len_s, ind, plot_x_ax=False, plot_y_ax=False):
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    for class_ind in plot_data.keys():
        time_ax = np.array(plot_data[class_ind])[:, 0] * hop_len_s
        y_ax = np.array(plot_data[class_ind])[:, ind]
        plot.plot(time_ax, y_ax, marker='.', color=cmap[class_ind], linestyle='None', markersize=4)
    plot.grid()
    plot.xlim([0, 60])
    if not plot_x_ax:
        plot.gca().axes.set_xticklabels([])

    if not plot_y_ax:
        plot.gca().axes.set_yticklabels([])


# --------------------------------- MAIN SCRIPT STARTS HERE -----------------------------------------
def visualize_output(params):
    feat_cls = feature_class.FeatureClass(params)

    sed_pred_reg = np.load('../results/sed_pred.npy')[0:600]
    sed_gt_reg = np.load('../results/sed_gt.npy')[0:600]
    doa_pred_reg = np.load('../results/doa_pred.npy')[0:600]
    doa_gt_reg = np.load('../results/doa_gt.npy')[0:600]

    pred_dict = feat_cls.regression_label_format_to_output_format(sed_pred_reg, doa_pred_reg)
    gt_dict = feat_cls.regression_label_format_to_output_format(sed_gt_reg, doa_gt_reg)
    pred_dict_polar = feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
    gt_dict_polar = feat_cls.convert_output_format_cartesian_to_polar(gt_dict)
    # output format file to visualize

    pred_data = collect_classwise_data(pred_dict_polar)
    ref_data = collect_classwise_data(gt_dict_polar)

    nb_classes = len(feat_cls.get_classes())

    # load the audio and extract spectrogram

    plot.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 4)
    ax1 = plot.subplot(gs[1, :2]), plot_func(ref_data, params.label_hop_len_s, ind=1, plot_y_ax=True), plot.ylim(
        [-1, nb_classes + 1]), plot.title('SED reference')
    ax2 = plot.subplot(gs[1, 2:]), plot_func(pred_data, params.label_hop_len_s, ind=1), plot.ylim(
        [-1, nb_classes + 1]), plot.title('SED predicted')
    ax3 = plot.subplot(gs[2, :2]), plot_func(ref_data, params.label_hop_len_s, ind=2, plot_y_ax=True), plot.ylim(
        [-180, 180]), plot.title('Azimuth reference')
    ax4 = plot.subplot(gs[2, 2:]), plot_func(pred_data, params.label_hop_len_s, ind=2), plot.ylim(
        [-180, 180]), plot.title('Azimuth predicted')
    ax5 = plot.subplot(gs[3, :2]), plot_func(ref_data, params.label_hop_len_s, ind=3, plot_y_ax=True), plot.ylim(
        [-90, 90]), plot.title('Elevation reference')
    ax6 = plot.subplot(gs[3, 2:]), plot_func(pred_data, params.label_hop_len_s, ind=3), plot.ylim([-90, 90]), plot.title(
        'Elevation predicted')
    ax_lst = [ax1, ax2, ax3, ax4, ax5, ax6]

    plot.savefig('../images/output.jpg', dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    params = parameter.get_params(output=False)
    visualize_output(params)

