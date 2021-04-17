import numpy as np


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :,
                                                                                        2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5

    return sed, accdoa_in