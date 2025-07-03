import cv2
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import metrics
import glob

from networks.Contrast_group.SNUNET.ModelLoss import hybrid_loss
from path import *
from tool_and_test.Customloss import dice_loss

from utils import save_pre_result
from visualization.visualizeConfusionMatrix import visualize_confusion_matrix

filename = glob.glob(test_src_t1 + '/*.png')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids = [0, 1]

def train(net, dataloader_train, total_step, criterion_ce, optimizer):
    print('Training...')
    model = net.train()

    # model_path = './ckps/customedNet1_4_continue/customedNet1_4_continue_batch=16_lr=0.0001_epoch18model.pth'
    # ckps = torch.load(model_path, map_location='cuda:0')
    # model.load_state_dict(ckps)

    num = 0 # 记录epoch次数
    epoch_loss = 0
    cm_total = np.zeros((2, 2))


    for x1, x2, y in dataloader_train:
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        inputs = torch.cat((inputs_t1, inputs_t2), dim=1)
        
        labels = y.to(device)

        optimizer.zero_grad()
        # pre1,pre2,pre= model(inputs) #pre为单通道预测图 USSFCNET
        pre,pre2= model(inputs)
        # pre= model(inputs_t1,inputs_t2) 
        # loss=criterion_ce(pre,labels)+0.4*criterion_ce(pre2,labels)

           # out_ch=1 只用了交叉熵损失
        # loss = criterion_ce(pre1, labels)+criterion_ce(pre2, labels)+2*criterion_ce(pre, labels)
        loss = criterion_ce(pre, labels)+0.4*criterion_ce(pre2,labels)
        # loss = criterion_ce(pre, labels) + dice_loss(pre, labels)  # out_ch=1 只用了交叉熵损失
       # loss = criterion_ce(pre, torch.squeeze(labels.long(), dim=1))   # out_ch=2

        loss.backward()
        # for name, parms in net.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #           ' -->grad_value:', parms.grad)


        epoch_loss += loss.item()
        optimizer.step()

        #pre = torch.max(pre, 1)[1]  # out_ch=2
        cm = metrics.ConfusionMatrix(2, pre, labels)
        cm_total += cm
        precision, recall, f1, iou, kc = metrics.get_score(cm)
        num += 1

        print('%d/%d, loss:%f, Pre:%f, Rec:%f, F1:%f, IoU:%f, KC:%f' % (num, total_step, loss.item(), precision[1], recall[1], f1[1], iou[1], kc))

    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)

    return epoch_loss, precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'], kc_total


def validate(net, dataloader_val, epoch):
    print('Validating...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))
    with torch.no_grad():
        for x1, x2, y in tqdm(dataloader_val):
            inputs_t1 = x1.to(device)
            inputs_t2 = x2.to(device)
            labels = y.to(device)
            
            inputs = torch.cat((inputs_t1, inputs_t2), dim=1)
            # pre1,pre2,pre = model(inputs)
            pre,pre2=model(inputs)
            # pre,pre2= model(inputs)
            # pre = model(inputs_t1,inputs_t2)
            # pre = torch.max(pre, 1)[1]  # out_ch=2
            cm = metrics.ConfusionMatrix(2, pre, labels)
            cm_total += cm
            num += 1
    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    return precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'], kc_total


# 可视化
path="G:/ExperimentResults/QualitativeComparisons/SYSU-CD/FC/predictvs"


import time

def predict(net, dataloader_test):
    print('Testing...')
    model = net.eval()
    num = 1
    cm_total = np.zeros((2, 2))

    total_infer_time = 0.0
    total_images = 0

    with torch.no_grad():
        for x1, x2, y in tqdm(dataloader_test):
            inputs_t1 = x1.to(device)
            inputs_t2 = x2.to(device)
            labels = y.to(device)
            inputs = torch.cat((inputs_t1, inputs_t2), dim=1)

            # Start timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            # Model inference
            pre,pre2 = model(inputs)

            # End timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            # Accumulate time
            infer_time = end_time - start_time
            total_infer_time += infer_time
            total_images += inputs.size(0)

            # Metrics
            cm = metrics.ConfusionMatrix(2, pre, labels)
            cm_total += cm

            # Optional: save results
            save_pre_result(pre, 'test', num, save_path=test_predict)
            #visualize_confusion_matrix(pre, labels, 'test_vs', num, cm, path)
            num += 1

    # Compute average inference time per image (in milliseconds)
    avg_infer_time_ms = (total_infer_time / total_images) * 1000
    print(f"\n[Inference Time] Total: {total_infer_time:.4f} s | Avg per image: {avg_infer_time_ms:.2f} ms")

    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    return precision_total, recall_total, f1_total, iou_total, kc_total

