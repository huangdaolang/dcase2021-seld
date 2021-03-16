# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
import argparse


def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--quick_test', type=bool, default=False,
                        help='To do quick test. Trains/test on small subset of dataset, and 2 epochs')
    parser.add_argument('--mode', type=str, default='dev',
                        help='dev or eval')
    parser.add_argument('--dataset', type=str, default='foa',
                        help='foa - ambisonic or mic - microphone signals')

    parser.add_argument('--input', type=str, default='mel',
                        help='determine which input format to use: mel or raw audio')
    parser.add_argument('--model', type=str, default='resnet',
                        help='resnet crnn or samplecnn')

    parser.add_argument('--dataset_dir', type=str, default='../Datasets/SELD2021/',
                        help='Base folder containing the foa/mic and metadata folders')
    parser.add_argument('--feat_label_dir', type=str, default='../Datasets/SELD2021/feat_label/',
                        help='Directory to dump extracted features and labels')
    parser.add_argument('--model_dir', type=str, default='trained_model/',
                        help='Dumps the trained models and training curves in this folder')
    parser.add_argument('--dcase_output', type=bool, default=True,
                        help='Set this true after you have finalized your model, save the output, and submit')
    parser.add_argument('--dcase_dir', type=str, default='results/',
                        help='Dumps the recording-wise network output in this folder')

    # FEATURE PARAMS
    parser.add_argument('--fs', type=int, default=24000)
    parser.add_argument('--hop_len_s', type=float, default=0.02)
    parser.add_argument('--label_hop_len_s', type=float, default=0.1)
    parser.add_argument('--max_audio_len_s', type=int, default=60)
    parser.add_argument('--nb_mel_bins', type=int, default=64)

    # DNN MODEL PARAMETERS
    parser.add_argument('--label_sequence_length', type=int, default=60,
                        help='Feature sequence length')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0,
                        help='Dropout rate, constant for all layers')
    parser.add_argument('--nb_cnn2d_filt', type=int, default=64,
                        help='Number of CNN nodes, constant for each layer')
    parser.add_argument('--f_pool_size', type=list, default=[4, 4, 2],
                        help='CNN frequency pooling, len[list] = number of CNN layers, list value = pooling per layer')
    parser.add_argument('--rnn_size', type=list, default=[128, 128],
                        help='RNN contents, length of list = number of layers, list value = number of nodes')
    parser.add_argument('--fnn_size', type=list, default=[128],
                        help='FNN contents, length of list = number of layers, list value = number of nodes')
    parser.add_argument('--loss_weights', type=list, default=[1., 1000.],
                        help='[sed, doa] weight for scaling the DNN outputs')

    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--epochs_per_fit', type=int, default=5,
                        help='Number of epochs per fit')
    parser.add_argument('--doa_objective', type=str, default='masked_mse',
                        help='supports: mse, masked_mse. mse- original seld approach; masked_mse - dcase 2020 approach')

    # METRIC PARAMETERS
    parser.add_argument('--lad_doa_thresh', type=int, default=20)

    parser.add_argument('--data_in', type=tuple, default=(7, 300, 64))
    parser.add_argument('--data_out', type=list, default=[(60, 14), (60, 56)])
    parser.add_argument('--scheduler', type=str, default='steplr')

    params = parser.parse_args()
    feature_label_resolution = int(params.label_hop_len_s // params.hop_len_s)
    params.feature_sequence_length = params.label_sequence_length * feature_label_resolution
    params.t_pool_size = [feature_label_resolution, 1, 1]  # CNN time pooling
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
    # print(type(params.__dict__))
    for key, value in params.__dict__.items():
        print("\t{}: {}".format(key, value))
    return params


if __name__ == '__main__':
    get_params()
