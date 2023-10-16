import argparse
import os
import shutil
import random
import cv2
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from Yolov3.yolov3.models import Darknet
from Yolov3.yolov3.utils.datasets import LoadImagesOnly,LoadTensor
from Yolov3.yolov3.utils.utils import (
    torch_utils,
    non_max_suppression,
    scale_coords,
    load_classes,
    #attempt_download,
)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def backtotensor(img0):
    # 假设img0是你想要还原回原始tensor的numpy数组

    # 逆向操作1：改变通道位置
    img0 = np.transpose(img0, (2, 0, 1))

    # 逆向操作2：将三个通道还原回一个通道
    img0 = np.mean(img0, axis=0, keepdims=True)

    # 逆向操作3：去除通道维度
    img0 = np.squeeze(img0)

    # 逆向操作4：将0-255还原回0-1范围
    img0 = img0 / 127.5 - 1
    img0 = torch.tensor(img0, dtype=torch.float32)
    return img0

    # 现在img0已经恢复成原始的tensor[0, self.count]格式

def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """Calculates the relative bounding box from absolute pixel values."""
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def _get_smoothborder_mask(height,width):
    # Calculate the center coordinates
    center_x = width / 2
    center_y = height / 2
    maxdistance = np.sqrt((width / 2)**2 + ( height / 2)**2)
    # Create an empty numpy array for the mask with shape (height, width)
    mask_img = np.zeros((height, width), dtype=np.float32)
    
    # Loop through each pixel and calculate its distance to the center
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Normalize the distance to the range [0, 1]
            normalized_distance = distance / maxdistance
            # Set the pixel value in the mask
            mask_img[y, x] = normalized_distance  # Invert the value to get closer to 1 at the center
    
    return mask_img

#一致性随机噪声+平滑
def draw_masks_SB(img, bbox, identities=None, offset=(0, 0),rand_center_x=0.5,rand_center_y=0.5,rand_width=0.5,rand_height=0.5):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(rand_center_x*(x2-x1))
        center_y = y1+int(rand_center_y*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(rand_height * bbox_height)
        mask_width = int(rand_width * bbox_height)

        # Calculate the top-left corner coordinates of the masking region
        mask_x1 = max(0,center_x - (mask_width // 2))
        mask_y1 = max(0,center_y - (mask_height // 2))
        mask_x2 = min(mask_x1 + mask_width,img.shape[1])
        mask_y2 = min(mask_y1 + mask_height,img.shape[0])
        
        new_height=mask_y2-mask_y1
        new_width=mask_x2-mask_x1

        #print(img.shape[0],img.shape[1])
        #print(mask_x1,mask_x2)
        #print(mask_y1,mask_y2)

        # Check if the region of interest is empty

        # Apply random noise to the masking region
        noise = np.random.rand(new_height, new_width, 3) * 255
        noise = noise.astype(np.uint8)

        mask_img=_get_smoothborder_mask(new_height, new_width)
        mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)

        #img_roi = cv2.addWeighted(img_roi, 0.5, mask, 0.5, 0)
        img[mask_y1:mask_y2, mask_x1:mask_x2] = mask_img*img[mask_y1:mask_y2, mask_x1:mask_x2]+(1-mask_img)*noise

    return img

#一致性随机纯色+平滑
def draw_masks_darkSB(img, bbox, identities=None, offset=(0, 0),rand_center_x=0.5,rand_center_y=0.5,rand_width=0.5,rand_height=0.5,randlight=0.5,cifar_img=None):
        
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(rand_center_x*(x2-x1))
        center_y = y1+int(rand_center_y*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(rand_height * bbox_height)
        mask_width = int(rand_width * bbox_height)

        # Calculate the top-left corner coordinates of the masking region
        mask_x1 = max(0,center_x - (mask_width // 2))
        mask_y1 = max(0,center_y - (mask_height // 2))
        mask_x2 = min(mask_x1 + mask_width,img.shape[1])
        mask_y2 = min(mask_y1 + mask_height,img.shape[0])
        
        new_height=mask_y2-mask_y1
        new_width=mask_x2-mask_x1

        if new_height>0 and new_width>0:
            if cifar_img is not None:
            
                cifar_pil_image = Image.fromarray(cifar_img)
                #print(new_height,new_width)
                cifar_pil_image = cifar_pil_image.resize((new_width, new_height), Image.BILINEAR)

                # 将 PIL 的 Image 对象转换回 NumPy 数组
                cifar_img_resized = np.array(cifar_pil_image)
                # Apply cifar_img to the masking region
                mask = cifar_img_resized
            
            else:
                # Apply dark to the masking region
                light = np.random.uniform(0, 0.5) 
                # Apply dark to the masking region
                mask = np.full([new_height, new_width, 3],light) * 255
                mask = mask.astype(np.uint8)

            mask_img=_get_smoothborder_mask(new_height, new_width)
            mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)

            img[mask_y1:mask_y2, mask_x1:mask_x2] = mask_img*img[mask_y1:mask_y2, mask_x1:mask_x2]+(1-mask_img)*mask

    return img

#随机纯色+平滑
def draw_masks_randbox_darkSB(img, bbox, identities=None, offset=(0, 0),cifar_img = None,return_rand = False):

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(np.random.uniform(0, 1)*(x2-x1))
        center_y = y1+int(np.random.uniform(0, 1)*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        rand_height = np.random.uniform(0, 1)
        rand_width = np.random.uniform(0, 1)
        mask_height = int(rand_height * bbox_height)
        mask_width = int( rand_width* bbox_height)
        # Calculate the top-left corner coordinates of the masking region

        mask_x1 = max(0,center_x - (mask_width // 2))
        mask_y1 = max(0,center_y - (mask_height // 2))
        mask_x2 = min(mask_x1 + mask_width,img.shape[1])
        mask_y2 = min(mask_y1 + mask_height,img.shape[0])
        
        new_height=mask_y2-mask_y1
        new_width=mask_x2-mask_x1

        if new_height>0 and new_width>0:
            if cifar_img is not None:
            
                cifar_pil_image = Image.fromarray(cifar_img)
                #print(new_height,new_width)
                cifar_pil_image = cifar_pil_image.resize((new_width, new_height), Image.BILINEAR)

                # 将 PIL 的 Image 对象转换回 NumPy 数组
                cifar_img_resized = np.array(cifar_pil_image)
                # Apply cifar_img to the masking region
                mask = cifar_img_resized
            
            else:
                # Apply dark to the masking region
                light = np.random.uniform(0, 1) 
                # Apply dark to the masking region
                mask = np.full([new_height, new_width, 3],light) * 255
                mask = mask.astype(np.uint8)

            mask_img=_get_smoothborder_mask(new_height, new_width)
            mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)

            img[mask_y1:mask_y2, mask_x1:mask_x2] = mask_img*img[mask_y1:mask_y2, mask_x1:mask_x2]+(1-mask_img)*mask
    if return_rand: 
        return img,rand_height+rand_width
    else:
        return img

#一致性随机纯色######################################
def draw_masks_dark(img, bbox, identities=None, offset=(0, 0),rand_center_x=0.5,rand_center_y=0.5,rand_width=0.5,rand_height=0.5,randlight=0.5,cifar_img=None):
        
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(rand_center_x*(x2-x1))
        center_y = y1+int(rand_center_y*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(rand_height * bbox_height)
        mask_width = int(rand_width * bbox_height)

        # Calculate the top-left corner coordinates of the masking region
        mask_x1 = max(0,center_x - (mask_width // 2))
        mask_y1 = max(0,center_y - (mask_height // 2))
        mask_x2 = min(mask_x1 + mask_width,img.shape[1])
        mask_y2 = min(mask_y1 + mask_height,img.shape[0])
        
        new_height=mask_y2-mask_y1
        new_width=mask_x2-mask_x1

        if new_height>0 and new_width>0:
            if cifar_img is not None:
            
                cifar_pil_image = Image.fromarray(cifar_img)
                #print(new_height,new_width)
                cifar_pil_image = cifar_pil_image.resize((new_width, new_height), Image.BILINEAR)

                # 将 PIL 的 Image 对象转换回 NumPy 数组
                cifar_img_resized = np.array(cifar_pil_image)
                # Apply cifar_img to the masking region
                mask = cifar_img_resized
            
            else:
                # Apply dark to the masking region
                light = np.random.uniform(0, 0.5) 
                # Apply dark to the masking region
                mask = np.full([new_height, new_width, 3],light) * 255
                mask = mask.astype(np.uint8)

            #mask_img=_get_smoothborder_mask(new_height, new_width)
            #mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)

            img[mask_y1:mask_y2, mask_x1:mask_x2] = mask

    return img

#随机噪声+平滑
def draw_masks_randboxSB(img, bbox, identities=None, offset=(0, 0)):

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(np.random.uniform(0, 1)*(x2-x1))
        center_y = y1+int(np.random.uniform(0, 1)*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(np.random.uniform(0, 1) * bbox_height)
        mask_width = int(np.random.uniform(0, 1) * bbox_height)
        # Calculate the top-left corner coordinates of the masking region

        mask_x1 = max(0,center_x - (mask_width // 2))
        mask_y1 = max(0,center_y - (mask_height // 2))
        mask_x2 = min(mask_x1 + mask_width,img.shape[1])
        mask_y2 = min(mask_y1 + mask_height,img.shape[0])
        
        new_height=mask_y2-mask_y1
        new_width=mask_x2-mask_x1

        if new_height>0 and new_width>0:

            # Apply dark to the masking region
            light = np.random.uniform(0, 0.5) 
            # Apply dark to the masking region
            mask = np.random.rand(new_height, new_width, 3) * 255
            mask = mask.astype(np.uint8)

            mask_img=_get_smoothborder_mask(new_height, new_width)
            mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)

            img[mask_y1:mask_y2, mask_x1:mask_x2] = mask_img*img[mask_y1:mask_y2, mask_x1:mask_x2]+(1-mask_img)*mask

    return img


#一致性随机噪声
def draw_masks(img, bbox, identities=None, offset=(0, 0),rand_center_x=0.5,rand_center_y=0.5,rand_width=0.5,rand_height=0.5):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(rand_center_x*(x2-x1))
        center_y = y1+int(rand_center_y*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(rand_height * bbox_height)
        mask_width = int(rand_width * bbox_height)

        # Calculate the top-left corner coordinates of the masking region
        mask_x1 = center_x - (mask_width // 2)
        mask_y1 = center_y - (mask_height // 2)
        mask_x2 = mask_x1 + mask_width
        mask_y2 = mask_y1 + mask_height

        # Check if the region of interest is empty

        # Apply random noise to the masking region
        mask = np.random.rand(mask_height, mask_width, 3) * 255
        mask = mask.astype(np.uint8)

        img_roi = img[mask_y1:mask_y2, mask_x1:mask_x2]
        if img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            mask = cv2.resize(mask, (img_roi.shape[1], img_roi.shape[0]))
            img_roi = cv2.addWeighted(img_roi, 0.5, mask, 0.5, 0)
            img[mask_y1:mask_y2, mask_x1:mask_x2] = img_roi

    return img

#随机噪声
def draw_masks_randbox(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(np.random.uniform(0, 1)*(x2-x1))
        center_y = y1+int(np.random.uniform(0, 1)*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(np.random.uniform(0, 1) * bbox_height)
        mask_width = int(np.random.uniform(0, 1) * bbox_height)

        # Calculate the top-left corner coordinates of the masking region
        mask_x1 = center_x - (mask_width // 2)
        mask_y1 = center_y - (mask_height // 2)
        mask_x2 = mask_x1 + mask_width
        mask_y2 = mask_y1 + mask_height

        # Check if the region of interest is empty

        # Apply random noise to the masking region
        mask = np.random.rand(mask_height, mask_width, 3) * 255
        mask = mask.astype(np.uint8)

        img_roi = img[mask_y1:mask_y2, mask_x1:mask_x2]
        if img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            mask = cv2.resize(mask, (img_roi.shape[1], img_roi.shape[0]))
            img_roi = cv2.addWeighted(img_roi, 0.5, mask, 0.5, 0)
            img[mask_y1:mask_y2, mask_x1:mask_x2] = img_roi

    return img

#随机纯色
def draw_masks_randbox_dark(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate center coordinates of the bounding box
        center_x = x1+int(np.random.uniform(0, 1)*(x2-x1))
        center_y = y1+int(np.random.uniform(0, 1)*(y2-y1))

        # Calculate the dimensions of the masking region
        bbox_height = y2 - y1
        mask_height = int(np.random.uniform(0, 1) * bbox_height)
        mask_width = int(np.random.uniform(0, 1) * bbox_height)

        # Calculate the top-left corner coordinates of the masking region
        mask_x1 = center_x - (mask_width // 2)
        mask_y1 = center_y - (mask_height // 2)
        mask_x2 = mask_x1 + mask_width
        mask_y2 = mask_y1 + mask_height

        # Check if the region of interest is empty
        light = np.random.uniform(0, 0.5) 
        # Apply random noise to the masking region
        mask = np.full([mask_height, mask_width, 3],light) * 255
        mask = mask.astype(np.uint8)

        img_roi = img[mask_y1:mask_y2, mask_x1:mask_x2]
        if img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            mask = cv2.resize(mask, (img_roi.shape[1], img_roi.shape[0]))
            img_roi = cv2.addWeighted(img_roi, 0.5, mask, 0.5, 0)
            img[mask_y1:mask_y2, mask_x1:mask_x2] = img_roi

    return img


def detect(model,
           dataset, save_img=True,
           names='Yolov3/yolov3/data/coco.names',
           output='output', conf_thres=0.3, iou_thres=0.5, half=False,
           device='0', save_txt=False, classes=[0],
           agnostic_nms=False,
           cifar_img = None,
           return_randsum= False):

    #img_size = (320, 192)  # if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    #webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    if cifar_img is not None:
        cifar_img = (cifar_img.permute(1, 2, 0).cpu()).numpy()
        cifar_img = (cifar_img * 255).astype(np.uint8)
        
        # Randomly crop a portion of cifar_img (between 0.1 to 1 of its size)
        cifar_height, cifar_width, _ = cifar_img.shape
        crop_factor = random.uniform(0.1, 1)
        crop_height = int(cifar_height * crop_factor)
        crop_width = int(cifar_width * crop_factor)
        crop_x1 = random.randint(0, cifar_width - crop_width)
        crop_y1 = random.randint(0, cifar_height - crop_height)
        cifar_img = cifar_img[crop_y1:crop_y1+crop_height, crop_x1:crop_x1+crop_width]

    # Initialize
    device = torch_utils.select_device(device=device)
    #if os.path.exists(output):
        #shutil.rmtree(output)  # delete output folder
    #os.makedirs(output)  # make new output folder
    rand_center_x=np.random.uniform(0, 1)
    rand_center_y=np.random.uniform(0, 1)
    rand_width=np.random.uniform(0.1, 1)
    rand_height=np.random.uniform(0.1, 1)
    light = np.random.uniform(0, 0.5) 
    rand_sum = 0

    if half:
        model.half()

    save_img = True
    names = load_classes(names)

    # Run inference
    t0 = time.time()
    num = 0
    tensor_list = []
    for img, im0s in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        #print('img.shape=',img.shape,'\n')
        #print(img.dtype)
        #print('im0s.shape=',im0s.shape,'\n')
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = '', '', im0s

            save_path = str(Path(output) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :5])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                # Write results
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape  # get image shape
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, bbox_left, bbox_top, bbox_w, bbox_h)
                    # print(x_c, y_c, bbox_w, bbox_h)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                
                #draw_masks_darkSB(im0, det[:, :4],rand_center_x=rand_center_x,rand_center_y=rand_center_y,rand_width=rand_width,rand_height=rand_height,randlight=light,cifar_img=cifar_img)  # 调用draw_masks函数
                #print('rand:',rand_height,rand_width)
                #rand_sum=rand_height+rand_width
                #draw_masks_randbox(im0, det[:, :4])  # 调用draw_masks函数
                #draw_masks_randbox_dark(im0, det[:, :4])
                #im0, randnum=draw_masks_randbox_darkSB(im0, det[:, :4],cifar_img=cifar_img,return_rand=True)
                draw_masks_dark(im0, det[:, :4],rand_center_x=rand_center_x,rand_center_y=rand_center_y,rand_width=rand_width,rand_height=rand_height,randlight=light,cifar_img=cifar_img)  # 调用draw_masks函数
                #rand_sum +=randnum


                #draw_masks(im0, det[:, :4],rand_center_x=rand_center_x,rand_center_y=rand_center_y,rand_width=rand_width,rand_height=rand_height)
              

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))
            # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                save_path += "/" + str(num) + '.jpg'
                num += 1
                #print(num)
                cv2.imwrite(save_path, im0)
        imgt = backtotensor(im0)
        tensor_list.append(imgt)
        #print("imgt.shape",imgt.shape,'\n')#torch.Size([256, 256]) 
    
        # 将列表中的每个 tensor 转换为张量并增加一个维度表示通道数
    
    tensor_list_as_tensors = [torch.unsqueeze(tensor, 0) for tensor in tensor_list]

    # 使用 torch.stack() 将所有张量沿着新添加的维度堆叠起来
    stacked_result = torch.stack(tensor_list_as_tensors, dim=1)

    # 打印结果张量的 shape
    #print(stacked_result.shape)  # 输出 torch.Size([1, 16, 256, 256])
    #if save_txt or save_img:
        #print('Results saved to %s' % os.getcwd() + os.sep + output)

    print('Done. (%.3fs)' % (time.time() - t0))
    if return_randsum:
        return stacked_result,rand_sum
    else:
        return stacked_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='Yolov3DeepSort/yolov3/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='Yolov3DeepSort/yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='Yolov3DeepSort/yolov3/weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='dataset/ped2/training/frames/01', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='2', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        dataset = LoadImagesOnly(opt.source, half=opt.half)
        detect(
            dataset=dataset,
            save_img=True,
            cfg=opt.cfg,
            names=opt.names,
            weights=opt.weights,
            output=opt.output,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            half=opt.half,
            device=opt.device,
            view_img=opt.view_img,
            save_txt=opt.save_txt,
            classes=opt.classes,
            agnostic_nms=opt.agnostic_nms
        )
