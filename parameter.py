import argparse


def get_params(output=True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--quick_test', type=int, default=0)
    parser.add_argument('--mode', type=str, default='dev',
                        help='dev or eval')
    parser.add_argument('--dataset', type=str, default='foa',
                        help='foa - ambisonic or mic - microphone signals')

    parser.add_argument('--input', type=str, default='raw',
                        help='determine which input format to use: mel or raw audio')
    parser.add_argument('--model', type=str, default='samplecnn',
                        help='if input==mel, choose resnet or crnn')
    parser.add_argument('--augmentation', type=int, default=1)
    parser.add_argument('--direct_read', type=int, default=0)

    parser.add_argument('--dataset_dir', type=str, default='../Datasets/SELD2020/',
                        help='Base folder containing the foa/mic and metadata folders')
    parser.add_argument('--feat_label_dir', type=str, default='../Datasets/SELD2020/feat_label/',
                        help='Directory to dump extracted features and labels')
    parser.add_argument('--model_dir', type=str, default='trained_model/',
                        help='Dumps the trained models and training curves in this folder')
    parser.add_argument('--dcase_output', type=bool, default=True,
                        help='Set this true after you have finalized your model, save the output, and submit')
    parser.add_argument('--dcase_dir', type=str, default='results/',
                        help='Dumps the recording-wise network output in this folder')

    # feature-based parameters
    parser.add_argument('--fs', type=int, default=24000)
    parser.add_argument('--hop_len_s', type=float, default=0.02)
    parser.add_argument('--label_hop_len_s', type=float, default=0.1)
    parser.add_argument('--max_audio_len_s', type=int, default=60)
    parser.add_argument('--nb_mel_bins', type=int, default=64)

    # model hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_epochs', type=int, default=40)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--mixup', type=int, default=0)

    # METRIC PARAMETERS
    parser.add_argument('--lad_doa_thresh', type=int, default=20)
    parser.add_argument('--label_sequence_length', type=int, default=60,
                        help='label length for one piece of data')
    parser.add_argument('--data_in', type=tuple, default=(7, 300, 64))
    parser.add_argument('--data_out', type=list, default=[(60, 14), (60, 42)])

    params = parser.parse_args()
    feature_label_resolution = int(params.label_hop_len_s // params.hop_len_s)
    params.feature_sequence_length = params.label_sequence_length * feature_label_resolution
    params.patience = int(params.nb_epochs)  # Stop training if patience is reached

    params.unique_classes = {
        'alarm': 0,
        'baby': 1,
        'crash': 2,
        'dog': 3,
        'engine': 4,
        'female_scream': 5,
        'female_speech': 6,
        'fire': 7,
        'footsteps': 8,
        'knock': 9,
        'male_scream': 10,
        'male_speech': 11,
        'phone': 12,
        'piano': 13
    }
    if output:
        for key, value in params.__dict__.items():
            print("\t{}: {}".format(key, value))
    return params


if __name__ == '__main__':
    get_params()
