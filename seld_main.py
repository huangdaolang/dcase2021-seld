import sys
import feature_class
import parameter
import data_loader
import solver
import datetime
import utils.utils_functions as utils


def main():
    # use parameter set defined by user
    params = parameter.get_params()

    feat_cls = feature_class.FeatureClass(params)
    train_splits, val_splits, test_splits = None, None, None

    if params.mode == 'dev':
        # dcase 2021 split
        test_splits = [6]
        val_splits = [6]
        train_splits = [[1, 2, 3, 4, 5]]
        # test_splits = [1]
        # val_splits = [2]
        # train_splits = [[3, 4, 5, 6]]

    elif params.mode == 'eval':
        test_splits = [[7, 8]]
        val_splits = [6]
        train_splits = [[1, 2, 3, 4, 5]]

    for split_cnt, split in enumerate(test_splits):
        utils.create_folder(params.model_dir)

        # Unique name for the run
        unique_name = '{}_{}'.format(
            datetime.datetime.today().strftime('%m%d%H%M'), params.model
        )
        print("unique_name: {}\n".format(unique_name))

        # Load train, validation and test data
        print('Loading training dataset:')
        if params.input == 'raw':
            data_train = data_loader.Tau_Nigens_raw(
                parameters=params, split=train_splits[split_cnt], is_val=False
            )

            print('Loading validation dataset:')
            data_val = data_loader.Tau_Nigens_raw(
                parameters=params, split=val_splits[split_cnt], is_val=True
            )

            print('Loading test dataset:')
            data_test = data_loader.Tau_Nigens_raw(
                parameters=params, split=test_splits[split_cnt], shuffle=False, is_val=True
            )
        else:
            data_train = data_loader.Tau_Nigens(
                parameters=params, split=train_splits[split_cnt], is_val=False
            )

            print('Loading validation dataset:')
            data_val = data_loader.Tau_Nigens(
                parameters=params, split=val_splits[split_cnt], is_val=True
            )

            print('Loading test dataset:')
            data_test = data_loader.Tau_Nigens(
                parameters=params, split=test_splits[split_cnt], shuffle=False, is_val=True
            )

        # create solver and run
        my_solver = solver.Solver(data_train, data_val, data_test, feat_cls, params, unique_name)
        my_solver.train()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)
