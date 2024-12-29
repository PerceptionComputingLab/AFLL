import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer import DefaultTrainer
import os
import numpy as np
from torch.utils.data import DataLoader
from utils.data_load.acne_dataset_processing import DatasetProcessing
from utils.transforms.affine_transforms import *


BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
NUM_WORKERS = 12

DATA_PATH = '/home/Jiangsonghan/data/Acne_Image_Grading/ACNE04/Classification/JPEGImages'

def main(args, cross_val_index):
    runseed = 42
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    TRAIN_FILE = '/home/zbf/data/Acne_Image_Grading/ACNE04/Classification/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = '/home/zbf/data/Acne_Image_Grading/ACNE04/Classification/NNEW_test_' + cross_val_index + '.txt'

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])


    dset_train = DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotate(rotation_range=20),
            normalize,
        ]))

    dset_test = DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_load = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_load = DataLoader(dset_test,
                             batch_size=BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

    trainer = DefaultTrainer(args)
    trainer.train(train_load, test_load)


if __name__ == '__main__':

    cfg = BaseConfig()
    loss_name = 'CELoss'
    alpha = 0.
    cross_val_lists = ['0', '1', '2', '3', '4']
    lamada = 1.0
    beta = 0.7
    for cross_val_index in cross_val_lists:
        model = 'AFLL'
        gpusss = 0
        ckpt_name = 'Rebuttal/{model}/Acne_4_lamada_{lamada}_beta_{beta}_CJH'.format(model=model, lamada=lamada, beta=beta)
        fixed = '--gpu_id {gpusss} ' \
                '--exp_name cross_val_{cross_val_index} ' \
                '--optim Adam ' \
                '--model_name {model} ' \
                '--lr 0.0001 ' \
                '--stepvalues 16000 ' \
                '--max_iter 1700 ' \
                '--loss_name {loss_name} ' \
                '--alpha {alpha} ' \
                '--lamada {lamada} ' \
                '--beta {beta} ' \
                '--warmup_steps 1 ' \
                '--val_freq 10 ' \
                '--num_classes 4 ' \
                '--save_folder /Share8/zbf/ord2seq_new/data/result/save_model/checkpoint_{ckpt_name}/ ' \
                '--save_log /Share8/zbf/ord2seq_new/data/result/save_log/logs_{ckpt_name}/ '.format(
            cross_val_index=cross_val_index,
            ckpt_name=ckpt_name,
            gpusss=gpusss,
            loss_name=loss_name,
            alpha=alpha,
            lamada=lamada,
            beta=beta,
            model=model) \
            .split()
        args = cfg.initialize(fixed)
        main(args, cross_val_index)