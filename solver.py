import torch
from torch.utils.data import DataLoader
from models import CRNN_mel, SampleCNN_raw, ResNet_mel, ReSE_SampleCNN
import torch.nn as nn
from torchsummary import summary
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR
from metrics import SELD_evaluation_metrics
from torch.utils.tensorboard import SummaryWriter
import os
import utils.utils_functions as utils
from torch_audiomentations import Compose, Shift, Gain
from transforms import Swap_Channel
import random
import torchaudio


class Solver(object):
    def __init__(self, data_train, data_val, data_test, feat_cls, params, unique_name):
        self.params = params
        self.feat_cls = feat_cls
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # model selection part
        if self.params.input == "mel":
            if self.params.model == "crnn":
                self.model = CRNN_mel.CRNN(dropout_rate=params.dropout_rate).to(self.device)
                summary(self.model, input_size=(7, 300, 64))
            elif self.params.model == "resnet":
                self.model = ResNet_mel.get_resnet(data_in=params.data_in, data_out=params.data_out).to(self.device)
                summary(self.model, input_size=(7, 300, 64))
        elif self.params.input == "raw":
            if self.params.model == "rese":
                self.model = ReSE_SampleCNN.ReSE_SampleCNN(params, ReSE_SampleCNN.Basic_Block).to(self.device)
            elif self.params.model == "samplecnn":
                self.model = SampleCNN_raw.SampleCNN(params).to(self.device)
            summary(self.model, input_size=(8, 144000))

        # optimizer selection part
        if self.params.optimizer == 'adam':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr, weight_decay=0.0001)
        elif self.params.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9)

        # scheduler selection part
        if self.params.scheduler == 'steplr':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif self.params.scheduler == 'cyclic':
            self.scheduler = CyclicLR(self.optimizer, base_lr=0.000001, max_lr=0.1, step_size_up=5, step_size_down=5)
        elif self.params.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=2, verbose=True)
        else:
            self.scheduler = None

        self.nb_epoch = 2 if self.params.quick_test == 1 else self.params.nb_epochs

        # mixup setup
        self.mixup = self.params.mixup
        self.alpha = 1.

        # augmentation setup
        self.augmentation = self.params.augmentation
        self.swap_channel = Swap_Channel()
        if self.augmentation == 1:
            self.apply_augmentation = Compose(
                transforms=[
                    Gain(
                        min_gain_in_db=-2.0,
                        max_gain_in_db=20.0,
                        p=0.5,
                    ),

                    # Shift(min_shift=-200, max_shift=200, shift_unit='samples')
                ]
            )
            self.masking = torchaudio.transforms.TimeMasking(100)

        # load data
        self.data_val = data_val
        self.data_train = data_train
        self.data_test = data_test
        self.train_dataloader = DataLoader(self.data_train, batch_size=params.batch_size, shuffle=True, drop_last=True)
        self.val_dataloader = DataLoader(self.data_val, batch_size=params.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.data_test, batch_size=params.batch_size, shuffle=False, drop_last=False)

        self.nb_classes = self.data_train.get_nb_classes()

        self.criterion = nn.MSELoss()
        self.best_seld_metric = 99999
        self.best_epoch = -1
        self.patience_cnt = 0

        self.early_stop_metric = np.zeros(self.nb_epoch)
        self.tr_loss = np.zeros(self.nb_epoch)
        self.seld_metric = np.zeros((self.nb_epoch, 4))

        self.lad_doa_thresh = self.params.lad_doa_thresh

        self.avg_scores_val = []
        self.unique_name = unique_name
        self.writer = SummaryWriter(os.path.join("log/", unique_name))

    def train(self):
        sed_pred = None
        sed_gt = None
        doa_pred = None
        doa_gt = None
        for epoch_cnt in range(self.nb_epoch):
            start = time.time()
            self.model.train()
            train_loss = 0
            for i, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                feature = data['feature'].to(self.device)
                label = data['label'].to(self.device)

                # data augmentation
                if self.augmentation == 1 and self.params.input == "raw":
                    p = random.random()
                    feature, label = self.swap_channel(feature, label, p)
                    feature = self.apply_augmentation(feature, sample_rate=24000)
                    feature = self.masking(feature)

                # mixup
                if self.mixup == 1:
                    feature, label_a, label_b, lam = utils.mixup_data(feature, label, self.alpha, self.use_cuda)
                    out = self.model(feature)
                    loss_func = utils.mixup_criterion(label_a, label_b, lam)
                    loss = loss_func(self.criterion, out)
                else:
                    out = self.model(feature)
                    loss = self.criterion(out, label)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.tr_loss[epoch_cnt] = train_loss / len(self.train_dataloader)
            print("Epoch [%d/%d], train loss : %.4f" % (epoch_cnt + 1, self.nb_epoch, self.tr_loss[epoch_cnt]))

            self.writer.add_scalar('Train Loss', self.tr_loss[epoch_cnt], epoch_cnt)
            self.writer.flush()

            # validate
            sed_out, doa_out, sed_label, doa_label, val_loss = self.validation(epoch_cnt)
            if self.params.scheduler == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(sed_out)
            doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(doa_out)
            sed_gt = SELD_evaluation_metrics.reshape_3Dto2D(sed_label)
            doa_gt = SELD_evaluation_metrics.reshape_3Dto2D(doa_label)

            # Calculate the DCASE 2020 metrics - Location-aware detection and Class-aware localization scores
            cls_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=self.nb_classes,
                                                             doa_threshold=self.lad_doa_thresh)
            pred_dict = self.feat_cls.regression_label_format_to_output_format(
                sed_pred, doa_pred
            )
            gt_dict = self.feat_cls.regression_label_format_to_output_format(
                sed_gt, doa_gt
            )

            pred_blocks_dict = self.feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
            gt_blocks_dict = self.feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

            cls_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
            self.seld_metric[epoch_cnt, :] = cls_metric.compute_seld_scores()
            self.early_stop_metric[epoch_cnt] = SELD_evaluation_metrics.early_stopping_metric(self.seld_metric[epoch_cnt, :2],
                                                                                         self.seld_metric[epoch_cnt, 2:])

            self.patience_cnt += 1
            if self.early_stop_metric[epoch_cnt] < self.best_seld_metric:
                self.best_seld_metric = self.early_stop_metric[epoch_cnt]
                self.best_epoch = epoch_cnt
                # model.save(model_name)
                self.patience_cnt = 0

            print(
                'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, '
                '\n\t\t DCASE2021 SCORES: SED_Error: {:0.2f}, SED_F: {:0.1f}, DOA_Error: {:0.1f}, DOA_recall:{:0.1f}, '
                'seld_score (early stopping score): {:0.2f}, '
                'best_seld_score: {:0.2f}, best_epoch : {}\n'.format(
                    epoch_cnt, time.time() - start, self.tr_loss[epoch_cnt],
                    self.seld_metric[epoch_cnt, 0], self.seld_metric[epoch_cnt, 1] * 100,
                    self.seld_metric[epoch_cnt, 2], self.seld_metric[epoch_cnt, 3] * 100,
                    self.early_stop_metric[epoch_cnt], self.best_seld_metric, self.best_epoch
                )
            )
            self.writer.add_scalar('ER', self.seld_metric[epoch_cnt, 0], epoch_cnt)
            self.writer.add_scalar('F', self.seld_metric[epoch_cnt, 1] * 100, epoch_cnt)
            self.writer.add_scalar('DE', self.seld_metric[epoch_cnt, 2], epoch_cnt)
            self.writer.add_scalar('DE_F', self.seld_metric[epoch_cnt, 3] * 100, epoch_cnt)
            self.writer.add_scalar('seld_score', self.early_stop_metric[epoch_cnt], epoch_cnt)
            self.writer.add_scalar('best seld_score', self.best_seld_metric, epoch_cnt)
            self.writer.flush()
            if self.patience_cnt > self.params.patience:
                break

        self.avg_scores_val.append([self.seld_metric[self.best_epoch, 0], self.seld_metric[self.best_epoch, 1],
                                    self.seld_metric[self.best_epoch, 2],
                                    self.seld_metric[self.best_epoch, 3], self.best_seld_metric])
        print('\nResults on validation split:')
        print('\tSaved model for the best_epoch: {}'.format(self.best_epoch))
        print('\tSELD_score (early stopping score) : {}'.format(self.best_seld_metric))

        print('\n\tDCASE2021 scores')
        print(
            '\tClass-aware localization scores: DOA_error: {:0.1f}, F-score: {:0.1f}'.format(
                self.seld_metric[self.best_epoch, 2],
                self.seld_metric[
                    self.best_epoch, 3] * 100))
        print(
            '\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(
                self.seld_metric[self.best_epoch, 0],
                self.seld_metric[
                    self.best_epoch, 1] * 100))

        # test set result
        self.test()

        # save model parameters
        model_name = self.unique_name + ".pth"
        torch.save(self.model.state_dict(), os.path.join(self.params.model_dir, model_name))

        # TODO add test set
        np.save('results/sed_pred.npy', sed_pred)
        np.save('results/sed_gt.npy', sed_gt)
        np.save('results/doa_pred.npy', doa_pred)
        np.save('results/doa_gt.npy', doa_gt)

    def validation(self, epoch_cnt):
        self.model.eval()
        sed_out = None
        doa_out = None
        sed_label = None
        doa_label = None
        val_loss = 0
        for i, data in enumerate(self.val_dataloader):
            feature = data['feature'].to(self.device)
            label = data['label'].to(self.device)

            out = self.model(feature)
            loss = self.criterion(out, label)
            val_loss += loss.item()
            sed_out_batch, doa_out_batch = utils.get_accdoa_labels(out.cpu().detach().numpy(), self.nb_classes)
            sed_label_batch, doa_label_batch = utils.get_accdoa_labels(label.cpu().detach().numpy(), self.nb_classes)
            if i == 0:
                sed_label = sed_label_batch
                doa_label = doa_label_batch
                sed_out = sed_out_batch
                doa_out = doa_out_batch
            else:
                sed_label = np.concatenate((sed_label, sed_label_batch), axis=0)
                doa_label = np.concatenate((doa_label, doa_label_batch), axis=0)
                sed_out = np.concatenate((sed_out, sed_out_batch), axis=0)
                doa_out = np.concatenate((doa_out, doa_out_batch), axis=0)

        print("Epoch [%d/%d], val loss : %.4f" % (epoch_cnt + 1, self.nb_epoch,
                                                  val_loss / len(self.val_dataloader)))
        val_loss = val_loss / len(self.val_dataloader)
        self.writer.add_scalar('Validation Loss', val_loss, epoch_cnt)
        self.writer.flush()

        return sed_out, doa_out, sed_label, doa_label, val_loss

    def test(self):
        print('\nLoading the best model and predicting results on the testing split')
        print('\tLoading testing dataset:')
        self.model.eval()
        sed_out = None
        doa_out = None
        sed_label = None
        doa_label = None
        for i, data in enumerate(self.test_dataloader):
            feature = data['feature'].to(self.device)
            label = data['label'].to(self.device)

            test_pred = self.model(feature)

            sed_out_batch, doa_out_batch = utils.get_accdoa_labels(test_pred.cpu().detach().numpy(), self.nb_classes)
            sed_label_batch, doa_label_batch = utils.get_accdoa_labels(label.cpu().detach().numpy(), self.nb_classes)

            if i == 0:
                sed_label = sed_label_batch
                doa_label = doa_label_batch
                sed_out = sed_out_batch
                doa_out = doa_out_batch
            else:
                sed_label = np.concatenate((sed_label, sed_label_batch), axis=0)
                doa_label = np.concatenate((doa_label, doa_label_batch), axis=0)
                sed_out = np.concatenate((sed_out, sed_out_batch), axis=0)
                doa_out = np.concatenate((doa_out, doa_out_batch), axis=0)

        sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(sed_out)
        doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(doa_out)
        sed_gt = SELD_evaluation_metrics.reshape_3Dto2D(sed_label)
        doa_gt = SELD_evaluation_metrics.reshape_3Dto2D(doa_label)

        cls_test_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=self.nb_classes,
                                                              doa_threshold=self.lad_doa_thresh)
        test_pred_dict = self.feat_cls.regression_label_format_to_output_format(
            sed_pred, doa_pred
        )
        test_gt_dict = self.feat_cls.regression_label_format_to_output_format(
            sed_gt, doa_gt
        )

        test_pred_blocks_dict = self.feat_cls.segment_labels(test_pred_dict, sed_pred.shape[0])
        test_gt_blocks_dict = self.feat_cls.segment_labels(test_gt_dict, sed_gt.shape[0])

        cls_test_metric.update_seld_scores_xyz(test_pred_blocks_dict, test_gt_blocks_dict)
        test_seld_metric = cls_test_metric.compute_seld_scores()
        test_early_stop_metric = SELD_evaluation_metrics.early_stopping_metric(test_seld_metric[:2],
                                                                               test_seld_metric[2:])

        print('Results on test split:')

        print('\tDCASE2021 Scores')
        print('\tClass-aware localization scores: DOA Error: {:0.1f}, F-score: {:0.1f}'.format(test_seld_metric[2],
                                                                                               test_seld_metric[
                                                                                                   3] * 100))
        print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(test_seld_metric[0],
                                                                                                test_seld_metric[
                                                                                                    1] * 100))
        print('\tSELD (early stopping metric): {:0.2f}'.format(test_early_stop_metric))
