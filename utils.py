import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import random


def get_bbox_loss(videoinput, videooutput, all_bboxes):
    # 获取stacked_result的形状
    channels, num_frames, height, width = videoinput.shape # torch.Size([1, 16, 256, 256])

    # 初始化损失
    loss = 0.0
    
    # 遍历每一帧
    for frame_idx in range(num_frames):       
        if(len(all_bboxes[frame_idx])>0):
            frame_bboxes = all_bboxes[frame_idx][0]  # 获取当前帧的所有bbox信息
            frame_output = videooutput[:, frame_idx, :, :]  # 获取模型输出的当前帧
            # 遍历当前帧的每个bbox
            for bbox_info in frame_bboxes:
                x_c, y_c, bbox_w, bbox_h = bbox_info

                # 计算bbox的左上角和右下角坐标
                x1 = int(x_c - bbox_w / 2)
                y1 = int(y_c - bbox_h / 2)
                x2 = int(x_c + bbox_w / 2)
                y2 = int(y_c + bbox_h / 2)

                # 提取目标框
                target_bbox = videoinput[:, frame_idx, y1:y2, x1:x2]

                # 提取模型输出的相应区域
                predicted_bbox = frame_output[:, y1:y2, x1:x2]

                # 使用均方差计算损失
                bbox_loss = F.mse_loss(predicted_bbox, target_bbox)
                
                # 将当前bbox的损失添加到总损失中
                loss += bbox_loss

    # 返回平均损失
    return loss / (len(all_bboxes) * num_frames)


def random_mask(video_tensor, ratio=0.1, n=10):
    _, num_frames, height, width = video_tensor.size()
    masked_tensor = video_tensor.clone()

    for frame_idx in range(num_frames):
        for _ in range(n):
            mask_height = random.randint(0, int(height * ratio))
            mask_width = random.randint(0, int(width * ratio))
            mask_x = random.randint(0, width - mask_width)
            mask_y = random.randint(0, height - mask_height)
            
            mask = torch.full((mask_height, mask_width), random.uniform(-1, 0))

            original_region = masked_tensor[0, frame_idx, mask_y:mask_y+mask_height, mask_x:mask_x+mask_width]
            combined_region = original_region * 0.5 + (mask * 0.5).cuda()

            masked_tensor[0, frame_idx, mask_y:mask_y+mask_height, mask_x:mask_x+mask_width] = combined_region

    return masked_tensor

# Example usage:
# Assuming you have a video tensor 'video' with shape (1, num_frames, height, width)
# masked_video = random_mask(video)



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def psnrv2(mse, peak):
    # note: peak = max(I) where I ranged from 0 to 2 (considering mse is calculated when I is ranged -1 to 1)
    return 10 * math.log10(peak * peak / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)  # +1/2 probably the g() function. Normalize from 0-1. Although not exactly min() and max() value.
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result
