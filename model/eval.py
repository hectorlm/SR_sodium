import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from lib.utils import fft_downsize
import os

def normlize_slice(im):
    im = (im - np.min(im)) / (np.percentile(im, 99.9) - np.min(im))
    return im

def compute_IoU(cm):
    '''
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    '''
    
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives
    
    iou = true_positives / denominator
    
    return iou, np.nanmean(iou) 


def eval_net_loader(net,dataset_detail, n_classes, batch_size, n_channel,normslice,criterion,device='gpu'):
    net.eval()
    train_num = dataset_detail['train_num']
    length_dataset = dataset_detail['length_dataset']
    train_path =dataset_detail['train_path']
    file_names_val=dataset_detail['file_names_val']
    acc = 0.0
    length = 0
    accf = 0.0
    lf = 0
    for index in range(0, max(batch_size, len(file_names_val) - batch_size), batch_size):
        ssize = min(len(file_names_val),batch_size)
        imgs = torch.empty(ssize, n_channel, 160, 160)
        true_masks = torch.empty(ssize, 160, 160)
        for i in range(0, ssize):
            with np.load(os.path.join(train_path, file_names_val[index + i])) as sample:
                if normslice:
                    imgs[i, :, :, :] = torch.from_numpy(normlize_slice(sample['lr']))
                else:
                    imgs[i, :, :, :] = torch.from_numpy(sample['lr'])
                true_masks[i, :, :] = torch.from_numpy(sample['hr'])

        imgs = imgs.to(device)
        true_masks = true_masks.to(device)  # torch.Size([4, 512, 512])

        outputs = net(imgs)
    #
        loss = criterion[0](outputs, torch.reshape(true_masks, [-1, 1, 160, 160]))
        loss2 = criterion[1](outputs, imgs[:, 0:1])
        acc += float(loss.cpu().detach())
        acc += float(loss2.cpu().detach())

        length += len(true_masks)
        accf += np.sum(np.abs((outputs-imgs).cpu().detach().numpy()))
        lf += len(imgs)
    return acc/length, accf/lf
