import torch
from torch.utils.data import DataLoader
from models import CRNN_mel, SampleCNN_raw, ResNet_mel
import torch.nn as nn
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR
from metrics import evaluation_metrics, SELD_evaluation_metrics
from torch.utils.tensorboard import SummaryWriter
import os


class Solver(object):
    def __init__(self, data_train, data_val, data_test, feat_cls, params, unique_name):
        self.params = params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.params.input == "mel":
            if self.params.model == "crnn":
                self.model = CRNN_mel.CRNN(data_in=params.data_in, data_out=params.data_out,
                                           dropout_rate=params.dropout_rate,
                                           nb_cnn2d_filt=params.nb_cnn2d_filt, f_pool_size=params.f_pool_size,
                                           t_pool_size=params.t_pool_size, rnn_size=params.rnn_size,
                                           fnn_size=params.fnn_size, doa_objective=params.doa_objective).to(self.device)
                self.model = self.model.double()
            elif self.params.model == "resnet":
                self.model = ResNet_mel.get_resnet(data_in=params.data_in, data_out=params.data_out).to(self.device)
                self.model = self.model.double()
        elif self.params.input == "raw":
            self.model = SampleCNN_raw.SampleCNN(params).to(self.device)
            self.model = self.model.double()

        self.optimizer = torch.optim.Adam(self.model.parameters())

        # scheduler
        if self.params.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=2, verbose=True)
        elif self.params.scheduler == 'cyclic':
            self.scheduler = CyclicLR(self.optimizer, base_lr=0.000001, max_lr=0.1, step_size_up=5, step_size_down=5)
        elif self.params.scheduler == 'steplr':
            self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.8)
        else:
            self.scheduler = None

        self.nb_epoch = 2 if self.params.quick_test else self.params.nb_epochs

        self.data_val = data_val
        self.data_train = data_train
        self.data_test = data_test
        self.val_dataloader = DataLoader(self.data_val, batch_size=params.batch_size, shuffle=True, drop_last=True)
        self.train_dataloader = DataLoader(self.data_train, batch_size=params.batch_size, shuffle=True,
                                           drop_last=True)
        # test set not implemented
        # self.test_dataloader = DataLoader(self.data_test, batch_size=params.batch_size, shuffle=False,
        #                                    drop_last=True)
        self.nb_classes = self.data_train.get_nb_classes()

        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.MSELoss()  # TODO add masked_mse
        self.best_seld_metric = 99999
        self.best_epoch = -1
        self.patience_cnt = 0

        self.early_stop_metric = np.zeros(self.nb_epoch)
        self.tr_loss = np.zeros(self.nb_epoch)
        self.seld_metric = np.zeros((self.nb_epoch, 4))
        self.feat_cls = feat_cls
        self.loss_weights = self.params.loss_weights
        self.lad_doa_thresh = self.params.lad_doa_thresh

        self.avg_scores_val = []
        self.unique_name = unique_name
        self.writer = SummaryWriter(os.path.join("log/", unique_name))

    def train(self):
        print("Model", self.model)
        # start training
        for epoch_cnt in range(self.nb_epoch):
            start = time.time()

            # train once per epoch
            self.model.train()
            train_loss = 0
            for i, data in enumerate(self.train_dataloader):
                feature = data['feature'].to(self.device)
                sed_label = data['label'][0].to(self.device)
                doa_label = data['label'][1].to(self.device)

                sed_out, doa_out = self.model(feature)
                loss = self.loss_weights[0] * self.criterion1(sed_out, sed_label) + \
                       self.loss_weights[1] * self.criterion2(doa_out, doa_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                train_loss += loss.item()
            print("Epoch [%d/%d], train loss : %.4f" % (epoch_cnt + 1, self.nb_epoch,
                                                        train_loss / len(self.train_dataloader)))
            self.writer.add_scalar('Train Loss', train_loss / len(self.train_dataloader), epoch_cnt)
            self.writer.flush()
            sed_out, doa_out, sed_label, doa_label = self.validation(epoch_cnt)

            sed_pred = evaluation_metrics.reshape_3Dto2D(sed_out) > 0.5
            sed_pred = sed_pred.cpu().detach().numpy()
            doa_pred = evaluation_metrics.reshape_3Dto2D(
                doa_out if self.params.doa_objective is 'mse' else doa_out[:, :,
                                                                   self.nb_classes:]).cpu().detach().numpy()
            sed_gt = evaluation_metrics.reshape_3Dto2D(sed_label.cpu()).detach().numpy()
            doa_gt = evaluation_metrics.reshape_3Dto2D(doa_label.cpu()).detach().numpy()[:, self.nb_classes:]

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
            self.early_stop_metric[epoch_cnt] = evaluation_metrics.early_stopping_metric(self.seld_metric[epoch_cnt, :2],
                                                                                         self.seld_metric[epoch_cnt, 2:])

            # Visualize the metrics with respect to epochs
            # plot_functions(unique_name, tr_loss, sed_metric, doa_metric, seld_metric, new_metric, new_seld_metric)

            self.patience_cnt += 1
            if self.early_stop_metric[epoch_cnt] < self.best_seld_metric:
                self.best_seld_metric = self.early_stop_metric[epoch_cnt]
                self.best_epoch = epoch_cnt
                # model.save(model_name)
                self.patience_cnt = 0

            print(
                'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, '
                '\n\t\t DCASE2020 SCORES: ER: {:0.2f}, F: {:0.1f}, DE: {:0.1f}, DE_F:{:0.1f}, '
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

    def validation(self, epoch_cnt):
        self.model.eval()
        sed_out = None
        doa_out = None
        sed_label = None
        doa_label = None
        val_loss = 0
        for i, data in enumerate(self.val_dataloader):
            feature = data['feature'].to(self.device)
            sed_label = data['label'][0].to(self.device)
            doa_label = data['label'][1].to(self.device)

            sed_out, doa_out = self.model(feature)
            loss = self.loss_weights[0] * self.criterion1(sed_out, sed_label) + \
                   self.loss_weights[1] * self.criterion2(doa_out, doa_label)
            val_loss += loss.item()

        print("Epoch [%d/%d], val loss : %.4f" % (epoch_cnt + 1, self.nb_epoch,
                                                  val_loss / len(self.val_dataloader)))
        self.writer.add_scalar('Validation Loss', val_loss / len(self.val_dataloader), epoch_cnt)
        self.writer.flush()
        return sed_out, doa_out, sed_label, doa_label

    def test(self):
        print('\nLoading the best model and predicting results on the testing split')
        print('\tLoading testing dataset:')
