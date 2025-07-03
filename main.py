import logging

import torchvision
from mmengine.optim import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import full_path_loader, CDDloader

from networks.EGDENetRevise import NewNetwork
from networks.netLoss import cd_loss

from operation import train, validate
from path import *
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from dataset import RsDataset
from tool_and_test.CustomTransforms import RandomApplyTransform
from tool_and_test.Customloss import FocalLoss
from tool_and_test.customedTrasforms import train_transforms, test_transforms
from utils import get_logger, random_seed



TITLE = ''

writer_train = SummaryWriter('runs/' + TITLE + '/train')
writer_val = SummaryWriter('runs/' + TITLE + '/val')
writer_all = SummaryWriter('runs/' + TITLE + '/all')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


src_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])

])


label_transform = transforms.Compose([
    transforms.ToTensor()
])





def main(args):

    random_seed(24)
    

    net = NewNetwork(in_ch=3, out_ch=1, ratio=0.5).to(device)
    start_epoch = 0
    total_epochs = 100  # df:100
    best_f1 = 0
    best_epoch = 0
    criterion_ce = nn.BCELoss()


    optimizer = optim.Adam(net.parameters(), args['lr'], weight_decay=0.00005)#  weight_decay=0.0005



    dataset_train = RsDataset(train_src_t1, train_src_t2, train_label,
                              t1_transform=src_transform,
                              t2_transform=src_transform,
                              label_transform=label_transform)

    dataset_val = RsDataset(test_src_t1, test_src_t2, test_label,
                            t1_transform=src_transform,
                            t2_transform=src_transform,
                            label_transform=label_transform)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  num_workers=12)

    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=12)


    num_dataset = len(dataloader_train.dataset)
    total_step = (num_dataset - 1) // dataloader_train.batch_size + 1

    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = get_logger('logs/' + TITLE + '.log')
    logger.info('Net: ' + TITLE)
    logger.info('Batch Size: {}'.format(args['batch_size']))
    logger.info('Learning Rate: {}'.format(args['lr']))

    ckp_savepath = 'ckps/' + TITLE
    if not os.path.exists(ckp_savepath):
        os.makedirs(ckp_savepath)

    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    for epoch in range(start_epoch, total_epochs):
        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('=' * 10)
        epoch += 1
        epoch_loss_train, pre_train, recall_train, f1_train, iou_train, kc_train = train(net, dataloader_train,
                                                                                         total_step, criterion_ce
                                                                                         , optimizer)
        scheduler.step()

        print('epoch %d - train loss:%f, train Pre:%f, train Rec:%f, train F1:%f, train iou:%f, train kc:%f' % (
            epoch, epoch_loss_train / total_step, pre_train, recall_train, f1_train, iou_train, kc_train))

        logger.info(
            'Epoch:[{}/{}]\t train_loss={:.5f}\t train_Pre={:.3f}\t train_Rec={:.3f}\t train_F1={:.3f}\t train_IoU={:.3f}\t train_KC={:.3f}'.format(
                epoch, total_epochs, epoch_loss_train / total_step, pre_train, recall_train, f1_train, iou_train,
                kc_train))
        writer_train.add_scalar('loss_of_train', epoch_loss_train / total_step, epoch)  # 训练集上，每个epoch的平均损失
        writer_train.add_scalar('f1_of_train', f1_train, epoch)
        writer_all.add_scalar('loss_of_train', epoch_loss_train / total_step, epoch)
        writer_all.add_scalar('f1_of_train', f1_train, epoch)

        pre_val, recall_val, f1_val, iou_val, kc_val = validate(net, dataloader_val, epoch)
        if f1_val > best_f1:

            best_f1 = f1_val
            best_epoch = epoch
            ckp_name = TITLE + '_batch={}_lr={}_epoch{}model.pth'.format(
                args['batch_size'],
                args['lr'],
                epoch)

            torch.save(net.state_dict(), os.path.join(ckp_savepath, ckp_name), _use_new_zipfile_serialization=False)

        print('epoch %d - val Pre:%f val Recall:%f val F1Score:%f' % (epoch, pre_val, recall_val, f1_val))
        logger.info(
            'Epoch:[{}/{}]\t val_Pre={:.4f}\t val_Rec:{:.4f}\t val_F1={:.4f}\t IoU={:.4f}\t KC={:.4f}\t best_F1:[{:.4f}/{}]\t'.format(
                epoch, total_epochs, pre_val, recall_val, f1_val, iou_val, kc_val, best_f1, best_epoch))

        writer_val.add_scalar('f1_of_validation', f1_val, epoch)
        writer_all.add_scalar('f1_of_validation', f1_val, epoch)
