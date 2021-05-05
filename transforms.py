import random
import torch
import feature_class
import numpy as np


class Swap_Channel(object):
    def __init__(self):
        self.p = random.random()

    def __call__(self, data, label):
        x = label[:, :, :14]
        y = label[:, :, 14:28]
        z = label[:, :, 28:]

        if 0 <= self.p < 0.125:
            data = data
            label = label
        elif 0.125 <= self.p < 0.250:
            data = torch.stack((data[:, 1], data[:, 3], data[:, 0], data[:, 2],
                                data[:, 4], -data[:, 7], -data[:, 6], data[:, 5]), dim=1)
            new_x = y
            new_y = -x
            new_z = -z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        elif 0.250 <= self.p < 0.375:
            data = torch.stack((data[:, 3], data[:, 1], data[:, 2], data[:, 0],
                                data[:, 4], -data[:, 7], data[:, 6], -data[:, 5]), dim=1)
            new_x = -y
            new_y = -x
            new_z = z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        elif 0.375 <= self.p < 0.500:
            data = torch.stack((data[:, 1], data[:, 0], data[:, 3], data[:, 2],
                                data[:, 4], -data[:, 5], -data[:, 6], data[:, 7]), dim=1)
            new_x = x
            new_y = -y
            new_z = -z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        elif 0.500 <= self.p < 0.625:
            data = torch.stack((data[:, 2], data[:, 0], data[:, 3], data[:, 1],
                                data[:, 4], data[:, 7], -data[:, 6], -data[:, 5]), dim=1)
            new_x = -y
            new_y = x
            new_z = -z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        elif 0.625 <= self.p < 0.750:
            data = torch.stack((data[:, 0], data[:, 2], data[:, 1], data[:, 3],
                                data[:, 4], data[:, 7], data[:, 6], data[:, 5]), dim=1)
            new_x = y
            new_y = x
            new_z = z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        elif 0.750 <= self.p < 0.875:
            data = torch.stack((data[:, 3], data[:, 2], data[:, 1], data[:, 0],
                                data[:, 4], -data[:, 5], data[:, 6], -data[:, 7]), dim=1)
            new_x = -x
            new_y = -y
            new_z = z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        elif 0.875 <= self.p < 1:
            data = torch.stack((data[:, 2], data[:, 3], data[:, 0], data[:, 1],
                                data[:, 4], data[:, 5], -data[:, 6], -data[:, 7]), dim=1)
            new_x = -x
            new_y = y
            new_z = -z
            label = torch.cat([new_x, new_y, new_z], dim=2)
        return data, label


if __name__ == "__main__":
    data = torch.tensor([
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]],
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]],
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]
    ])

    swap_channel = Swap_Channel()
    tmp_label_1 = np.load("../Datasets/SELD2020/feat_label/foa_dev_label/fold1_room1_mix010_ov1.npy")[:10, 14:]

    tmp_label_2 = np.load("../Datasets/SELD2020/feat_label/foa_dev_label/fold1_room1_mix012_ov1.npy")[184:194, 14:]

    tmp_label = np.stack([tmp_label_1, tmp_label_2], axis=0)
    print(tmp_label.shape)
    data, label = swap_channel(data, torch.tensor(tmp_label))
    # print(data)
    print(data.shape)
    print(label.shape)
