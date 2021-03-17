import os
import sys
import feature_class
import parameter
import data_loader
import solver
import datetime


def main():
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1
    """

    # use parameter set defined by user
    params = parameter.get_params()

    feat_cls = feature_class.FeatureClass(params)
    train_splits, val_splits, test_splits = None, None, None

    if params.mode == 'dev':
        test_splits = [6]
        val_splits = [5]
        train_splits = [[1, 2, 3, 4]]

    elif params.mode == 'eval':
        test_splits = [[7, 8]]
        val_splits = [[6]]
        train_splits = [[1, 2, 3, 4, 5]]

    for split_cnt, split in enumerate(test_splits):

        # Unique name for the run
        feature_class.create_folder(params.model_dir)
        unique_name = '{}_{}_{}_split{}'.format(
            params.model, params.dataset, params.mode, split
        )

        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_train = data_loader.Tau_nigens(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_val = data_loader.Tau_nigens(
            params=params, split=val_splits[split_cnt]
        )

        print('Loading test dataset:')
        data_test = data_loader.Tau_nigens(
            params=params, split=test_splits[split_cnt]
        )

        print(
            'MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n\tdoa_objective: {}\n'.format(
                params.dropout_rate, params.nb_cnn2d_filt, params.f_pool_size, params.t_pool_size,
                params.rnn_size,
                params.fnn_size, params.doa_objective))

        print('Using loss weights : {}'.format(params.loss_weights))

        # create solver and run
        my_solver = solver.Solver(data_train, data_val, data_test, feat_cls, params, unique_name)
        my_solver.train()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)
