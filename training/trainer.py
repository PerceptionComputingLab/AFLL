import os, sys
import cv2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import models
from datetime import datetime
from tensorboardX import SummaryWriter
import copy
from config.cfg import arg2str
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from evaluater import metric
from evaluater.report import report_precision_se_sp_yi
from matplotlib import pyplot as plt
import scienceplots
from sklearn.manifold import TSNE

plt.style.use('science')

def is_fc(para_name):
    split_name = para_name.split('.')
    if split_name[-2] == 'final':
        return True
    else:
        return False


class DefaultTrainer(object):

    def __init__(self, args):
        self.pred_error = pd.DataFrame()
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.warmup_steps = args.warmup_steps
        self.eval_only = args.eval_only
        self.model = getattr(models, args.model_name.lower())(args)
        self.model.cuda()
        self.loss = nn.CrossEntropyLoss()
        self.max_acc = 0
        self.max_Pre = 0
        self.max_SE = 0
        self.max_SP = 0
        self.max_YI = 0
        self.tmp_idx_acc_with_mae = 0
        self.tmp_idx_acc_with_mae = 0
        self.min_loss = 1000
        self.min_mae = 1000
        self.loss_name = args.loss_name
        self.start = 0
        self.wrong = None
        self.log_path = os.path.join(self.args.save_folder, self.args.exp_name, 'result.txt')
        # self.log = open(self.log_path, mode='w')
        # self.log.write('============ ACC with MAE ============\n')
        # self.log.close()

        if args.loss_name != 'POE':
            if self.args.optim == 'Adam':
                self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                              betas=(0.9, 0.999), eps=1e-08)
            else:
                self.optim = getattr(torch.optim, args.optim) \
                    (filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=args.weight_decay)
        else:
            # 这个只是用于vgg2的
            # print('LR = 0.0001')
            params = []
            for keys, param_value in self.model.named_parameters():
                if (is_fc(keys)):
                    params += [{'params': [param_value], 'lr': 0.001}]
                else:
                    params += [{'params': [param_value], 'lr': 0.0001}]
            #
            self.optim = torch.optim.Adam(params, lr=self.lr,
                                          betas=(0.9, 0.999), eps=1e-08)
        #
        # if args.resume:
        #     if os.path.isfile(self.args.resume):
        #         iter, index = self.load_model(args.resume)
        #         self.start_iter = iter
        # state_dict = '/home/Jiangsonghan/ord2seq_new/data/result/save_model/checkpoint_Rebuttal/ldl_cjs/Acne_cross_9/cross_val_0/pvt_best_acc_0.8835616438356164.pth'
        # state_dict = torch.load(state_dict)
        # self.model.load_state_dict(state_dict['net_state_dict'])
        # for name, parameter in self.model.named_parameters():
        #     if "grade" not in name:
        #         parameter.requires_grad = False

    def train_iter(self, step, dataloader):
        # self.optim.zero_grad()
        img, label, lesion = next(dataloader)#dataloader.next()
        img = img.float().cuda()
        label = label.int().cuda()
        lesion = lesion.numpy()


        self.model.train()
        if self.eval_only:
            self.model.eval()

        pred, grade, loss = self.model(img, label)
        # loss = self.loss(pred, label)

        '''generate logger'''
        if self.start == 0:
            self.init_writer()
            self.start = 1

        print('Training - Step: {} - Loss: {:.4f}' \
              .format(step, loss.item()))

        loss.backward()
        self.optim.step()
        # self._adjust_learning_rate_iter(step)
        self.model.zero_grad()

        if step % self.args.display_freq == 0:
            if self.loss_name == 'POE':
                # pred是一个序列
                acc, mae = metric.cal_mae_acc_cls(pred, label)
            else:
                # acc = metric.accuracy(pred, label)
                # mae = metric.MAE(pred, label)
                acc = metric.accuracy(torch.softmax(pred, dim=1) + grade, label)
                mae = metric.MAE(torch.softmax(pred, dim=1) + grade, label)

            print(
                'Training - Step: {} - Acc: {:.4f} - MAE {:.4f} - lr:{}' \
                    .format(step, acc, mae, self.lr_current))

            # scalars = [loss.item(), acc, prec, recall, f1, kap]
            # names = ['loss', 'acc', 'precision', 'recall', 'f1score', 'kappa']
            scalars = [loss.item(),  acc, mae, self.lr_current]
            names = ['loss1', 'acc', 'MAE', 'lr']
            write_scalars(self.writer, scalars, names, step, 'train')

    def train(self, train_dataloader, valid_dataloader=None):

        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)

        for step in range(self.start_iter, self.max_iter):

            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                             train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (
                    step % self.args.val_freq == 0 or step == self.args.max_iter - 1) and (step != 0):
                val_iter = iter(valid_dataloader)
                val_loss, val_acc, val_Pre, val_SE, val_SP, val_YI = self.validation(step, val_iter, val_epoch_size)
                if val_acc > self.max_acc:
                    self.delete_model(best='best_acc', index=self.max_acc)
                    self.max_acc = val_acc
                    self.save_model(step, best='best_acc', index=self.max_acc, gpus=1)

                    self.log = open(self.log_path, mode='a')
                    self.log.write('best_acc_with_Pre_SE_SP_YI = {}\n'.format([val_acc, val_Pre, val_SE, val_SP, val_YI]))
                    self.log.close()

                if val_loss < self.min_loss:
                    self.delete_model(best='min_loss', index=self.min_loss)
                    self.min_loss = val_loss.item()
                    self.save_model(step, best='min_loss', index=self.min_loss, gpus=1)

                if val_Pre > self.max_Pre:
                    self.delete_model(best='best_Pre', index=self.max_Pre)
                    self.max_Pre = val_Pre
                    self.save_model(step, best='best_Pre', index=self.max_Pre, gpus=1)

                    self.log = open(self.log_path, mode='a')
                    self.log.write('best_Pre_with_acc_SE_SP_YI = {}\n'.format([val_Pre, val_acc, val_SE, val_SP, val_YI]))
                    self.log.close()

                if val_SE > self.max_SE:
                    self.delete_model(best='best_SE', index=self.max_SE)
                    self.max_SE = val_SE
                    self.save_model(step, best='best_SE', index=self.max_SE, gpus=1)

                    self.log = open(self.log_path, mode='a')
                    self.log.write('best_SE_with_acc_Pre_SP_YI = {}\n'.format([val_SE, val_acc, val_Pre, val_SP, val_YI]))
                    self.log.close()

                if val_SP > self.max_SP:
                    self.delete_model(best='best_SP', index=self.max_SP)
                    self.max_SP = val_SP
                    self.save_model(step, best='best_SP', index=self.max_SP, gpus=1)

                    self.log = open(self.log_path, mode='a')
                    self.log.write('best_SP_with_acc_Pre_SE_YI = {}\n'.format([val_SP, val_acc, val_Pre, val_SE, val_YI]))
                    self.log.close()

                if val_YI > self.max_YI:
                    self.delete_model(best='best_YI', index=self.max_YI)
                    self.max_YI = val_YI
                    self.save_model(step, best='best_YI', index=self.max_YI, gpus=1)

                    self.log = open(self.log_path, mode='a')
                    self.log.write('best_YI_with_acc_Pre_SE_SP = {}\n'.format([val_YI, val_acc, val_Pre, val_SE, val_SP]))
                    self.log.close()

        return self.min_loss, self.max_acc, self.max_Pre, self.max_SE, self.max_SP, self.max_YI

    def validation(self, step, val_iter, val_epoch_size):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()

        total_score = []
        total_target = []
        total_grade = []
        loss_t = 0
        with torch.no_grad():
            for i in range(val_epoch_size):

                img, target, lesion = next(val_iter)
                img = img.float().cuda()
                target = target.int().cuda()
                lesion = lesion.numpy()

                score, grade, loss = self.model(img, copy.deepcopy(target))
                loss_t = loss_t + loss
                if i == 0:
                    total_score = score
                    total_target = target
                    total_grade = grade
                else:
                    if len(score.shape) == 1:
                        score = score.unsqueeze(0)
                    if self.loss_name == 'POE':
                        total_score = torch.cat((total_score, score), 1)
                    else:
                        total_grade = torch.cat((total_grade, grade), 0)
                        total_score = torch.cat((total_score, score), 0)
                    total_target = torch.cat((total_target, target), 0)

        # loss = self.loss(total_score, total_target)
        if self.loss_name == 'POE':
            acc, mae = metric.cal_mae_acc_cls(total_score, total_target)
        else:
            y_pred = (torch.softmax(total_score, dim=1) + total_grade).max(1)[1].int().cpu().data.numpy()
            groundtrue = total_target.cpu().data.numpy()
            Result,acc,report = report_precision_se_sp_yi(y_pred, groundtrue)

        '''
        记录做错的img
        '''
        self.wrong_perspective_target = total_target.cpu().numpy()
        _, pred = total_score.max(1)
        wrong = (pred != total_target).float()
        wrong = wrong.cpu().numpy()
        wrong_idx = np.where(wrong == 1)[0]
        pred_wrong = pred[wrong_idx].cpu().numpy()
        # 预测错误的实际的label
        wrong_true = self.wrong_perspective_target[wrong_idx]
        if self.pred_error.empty:
            # self.wrong = wrong
            self.pred_error = pd.DataFrame({"step":step,"pred":[pred_wrong],"target":[wrong_true]})
        else:
            # self.wrong += wrong
            self.pred_error = self.pred_error._append(pd.DataFrame([[step,pred_wrong,wrong_true]], columns=self.pred_error.columns))


        print(
            'Valid - Step: {} \n Loss: {:.4f} \n Acc: {:.4f} \n Pre: {:.4f} \n SE: {:.4f} \n SP: {:.4f} \n YI: {:.4f}' \
                .format(step, loss_t.item(), acc, Result[0], Result[1], Result[2], Result[3]))

        scalars = [loss_t.item(), acc, Result[0], Result[1], Result[2], Result[3]]
        names = ['loss1', 'acc', 'Pre', "SE", 'SP', 'YI']
        write_scalars(self.writer, scalars, names, step, 'val')

        return loss_t, acc, Result[0], Result[1], Result[2], Result[3]

    def log_pred(self, val_iter, val_epoch_size):
        print("========Begin Visualization=============:")
        self.model.load_state_dict(torch.load(
            "/home/zbf/AFLL/data/result/save_model/checkpoint_Rebuttal/main/Acne_1_lamada_1.0/cross_val_1/pvt_best_acc_0.9075342465753424.pth")[
                                       'net_state_dict'])
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for i in range(val_epoch_size):
                img, label, lesion = next(val_iter)
                img = img.float().cuda()
                label = label.int().cuda()
                score, grade, loss = self.model(img, label)
                feature = torch.softmax(score, dim=1) + grade
                # feature = pred + grade
                preds.append(feature.cpu().data.numpy())
                labels.append(label.cpu().data.numpy())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            preds = [np.argmax(arr) for arr in preds]
            labels = labels.tolist()
            # 转化为numpy数组
            preds = np.array(preds)
            labels = np.array(labels)

            # 筛选出 label 为 0 的情况
            pred_when_target_is_0 = preds[labels == 0]
            # 计算不同预测值的概率
            prob_pred_0 = np.sum(pred_when_target_is_0 == 0) / len(pred_when_target_is_0)
            prob_pred_1 = np.sum(pred_when_target_is_0 == 1) / len(pred_when_target_is_0)
            prob_pred_other = np.sum((pred_when_target_is_0 != 0) & (pred_when_target_is_0 != 1)) / len(
                pred_when_target_is_0)

            print(f"真值为0时，预测正确的概率: {prob_pred_0:.2f}")
            print(f"真值为0时，预测为相邻类别的概率: {prob_pred_1:.2f}")
            print(f"真值为0时，预测值为其他的概率: {prob_pred_other:.2f}")

            # 筛选出 label 为 1 的情况
            pred_when_target_is_1 = preds[labels == 1]
            # 计算不同预测值的概率
            prob_pred_1_when_target_is_1 = np.sum(pred_when_target_is_1 == 1) / len(pred_when_target_is_1)
            prob_pred_0_or_2 = np.sum((pred_when_target_is_1 == 0) | (pred_when_target_is_1 == 2)) / len(
                pred_when_target_is_1)
            prob_pred_other = np.sum(
                (pred_when_target_is_1 != 0) & (pred_when_target_is_1 != 1) & (pred_when_target_is_1 != 2)) / len(
                pred_when_target_is_1)

            print(f"真值为1时，预测正确的概率: {prob_pred_1_when_target_is_1:.2f}")
            print(f"真值为1时，预测为相邻类别的概率: {prob_pred_0_or_2:.2f}")
            print(f"真值为1时，预测值为其他的概率: {prob_pred_other:.2f}")

            # 筛选出 label 为 2 的情况
            pred_when_target_is_2 = preds[labels == 2]
            # 计算不同预测值的概率
            prob_pred_2_when_target_is_2 = np.sum(pred_when_target_is_2 == 2) / len(pred_when_target_is_2)
            prob_pred_1_or_3 = np.sum((pred_when_target_is_2 == 1) | (pred_when_target_is_2 == 3)) / len(
                pred_when_target_is_2)
            prob_pred_other = np.sum(
                (pred_when_target_is_2 != 3) & (pred_when_target_is_2 != 1) & (pred_when_target_is_2 != 2)) / len(
                pred_when_target_is_2)

            print(f"真值为2时，预测正确的概率: {prob_pred_2_when_target_is_2:.2f}")
            print(f"真值为2时，预测为相邻类别的概率: {prob_pred_1_or_3:.2f}")
            print(f"真值为2时，预测为其他的概率: {prob_pred_other:.2f}")

            # 筛选出 label 为 3 的情况
            pred_when_target_is_3 = preds[labels == 3]
            # 计算不同预测值的概率
            prob_pred_3 = np.sum(pred_when_target_is_3 == 3) / len(pred_when_target_is_3)
            prob_pred_2 = np.sum(pred_when_target_is_3 == 2) / len(pred_when_target_is_3)
            prob_pred_other = np.sum((pred_when_target_is_3 != 2) & (pred_when_target_is_3 != 3)) / len(
                pred_when_target_is_3)

            print(f"真值为3时，预测正确的概率: {prob_pred_3:.2f}")
            print(f"真值为3时，预测为相邻类别的概率: {prob_pred_2:.2f}")
            print(f"真值为3时，预测值为其他的概率: {prob_pred_other:.2f}")



        print("success")


    def t_SNE(self, val_iter, val_epoch_size):
        print("========Begin Visualization=============:")
        self.model.load_state_dict(torch.load("/home/zbf/AFLL/data/result/save_model/checkpoint_Rebuttal/main/Acne_1_lamada_1.0/cross_val_1/pvt_best_acc_0.9075342465753424.pth")['net_state_dict'])
        self.model.eval()

        features = []
        labels = []
        with torch.no_grad():
            for i in range(val_epoch_size):
                img, label, lesion = next(val_iter)
                img = img.float().cuda()
                label = label.int().cuda()
                pred, grade, loss = self.model(img, label)
                feature = torch.softmax(pred, dim=1) + grade
                # feature = pred + grade
                features.append(feature.cpu().data.numpy())
                labels.append(label.cpu().data.numpy())
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
            features = features.reshape(len(features), -1)
            tsne = TSNE(n_components=2)
            tsne_result = tsne.fit_transform(features)
            clist = ['#00C9A7', '#FFC75F', '#F3C5FF', '#845EC2']
            newcmp = LinearSegmentedColormap.from_list('chaos', clist)
            with plt.style.context(['science', 'ieee', 'no-latex']):
                plt.figure(figsize=(10, 8))
                plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=newcmp, s=40, edgecolors='none')
                # plt.colorbar()
                plt.axis('off')
                # plt.title("t-SNE visualization")
                # plt.xlabel('t-SNE Dimension 1')
                # plt.ylabel('t-SNE Dimension 2')
                plt.savefig('/home/zbf/AFLL/pic/my.jpg', dpi=300)
                plt.show()


    def test(self, val_iter, val_epoch_size):

        print('============Begin Validation============:')

        self.model.eval()

        total_score = []
        total_target = []
        total_grade = []
        loss_t = 0
        with torch.no_grad():
            for i in range(val_epoch_size):

                img, target, lesion = next(val_iter)
                img = img.float().cuda()
                target = target.int().cuda()
                lesion = lesion.numpy()

                score, grade, loss = self.model(img, copy.deepcopy(target))
                # loss_t += sum(loss)
                loss_t = loss_t + loss
                if i == 0:
                    total_score = score
                    total_target = target
                    total_grade = grade
                else:
                    if len(score.shape) == 1:
                        score = score.unsqueeze(0)
                    if self.loss_name == 'POE':
                        total_score = torch.cat((total_score, score), 1)
                    else:
                        total_grade = torch.cat((total_grade, grade), 0)
                        total_score = torch.cat((total_score, score), 0)
                    total_target = torch.cat((total_target, target), 0)

        # loss = self.loss(total_score, total_target)
        if self.loss_name == 'POE':
            acc, mae = metric.cal_mae_acc_cls(total_score, total_target)
        else:
            # acc = metric.accuracy(total_score, total_target)
            # mae = metric.MAE(total_score, total_target)
            y_pred = (torch.softmax(torch.sigmoid(total_score), dim=1) + total_grade).max(1)[1].int().cpu().data.numpy()
            groundtrue = total_target.cpu().data.numpy()
            Result, acc, report = report_precision_se_sp_yi(y_pred, groundtrue)
        print(
            'Valid - Loss: {:.4f} \n Acc: {:.4f} \n Pre: {:.4f} \n SE: {:.4f} \n SP: {:.4f} \n YI: {:.4f}' \
                .format( loss_t.item(), acc, Result[0], Result[1], Result[2], Result[3]))

    def log_wrong(self):
        self.pred_error.to_csv("/home/zbf/AFLL/data/result/pred_error/{modelName}_pred_error.csv".format(modelName=self.args.model_name.lower()),index=None)
        pass

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if step <= self.warmup_steps:  # 增大学习率
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        """ Tensorboard writer initialization
            """

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)

        if self.args.exp_name == 'test':
            log_path = os.path.join(self.args.save_log, self.args.exp_name)
        else:
            log_path = os.path.join(self.args.save_log,
                                    datetime.now().strftime('%b%d_%H-%M-%S') + '_' + self.args.optim + '_' + self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')

        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))

    def load_model(self, model_path):
        if os.path.exists(model_path):
            load_dict = torch.load(model_path)
            net_state_dict = load_dict['net_state_dict']

            try:
                self.model.load_state_dict(net_state_dict)
            except:
                self.model.module.load_state_dict(net_state_dict)
            self.iter = load_dict['iter'] + 1
            index = load_dict['index']

            print('Model Loaded!')
            return self.iter, index
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        if os.path.exists(save_path):
            os.remove(save_path)

    def save_model(self, step, best='best_acc', index=None, gpus=1):

        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if gpus == 1:
            if isinstance(index, list):
                save_fname = '%s_%s_%s_%s.pth' % (self.model.model_name(), best, index[0], index[1])
            else:
                save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        else:
            save_fname = '%s_%s_%s.pth' % (self.model.module.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.module.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')


def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)

if __name__ == '__main__':
    wrong = np.load("/home/zbf/AFLL/filename.npy")
    print(wrong)
