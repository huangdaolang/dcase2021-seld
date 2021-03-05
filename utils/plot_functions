import matplotlib.pyplot as plot


def plot_functions(fig_name, _tr_loss, _sed_loss, _doa_loss, _epoch_metric_loss, _new_metric, _new_seld_metric):
    plot.switch_backend('agg')

    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(411)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(412)
    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='sed er')
    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='sed f1')
    plot.plot(range(nb_epoch), _doa_loss[:, 0] / 180., label='doa er / 180')
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='doa fr')
    plot.plot(range(nb_epoch), _epoch_metric_loss, label='seld')
    plot.legend()
    plot.grid(True)

    plot.subplot(413)
    plot.plot(range(nb_epoch), _new_metric[:, 0], label='seld er')
    plot.plot(range(nb_epoch), _new_metric[:, 1], label='seld f1')
    plot.plot(range(nb_epoch), _new_metric[:, 2] / 180., label='doa er / 180')
    plot.plot(range(nb_epoch), _new_metric[:, 3], label='doa fr')
    plot.plot(range(nb_epoch), _new_seld_metric, label='seld')

    plot.legend()
    plot.grid(True)

    plot.subplot(414)
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_pks')
    plot.plot(range(nb_epoch), _doa_loss[:, 3], label='good_pks')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()
