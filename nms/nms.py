#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:43:28 2022

@author: sagar
"""
import torch

###################### USING CUSTOM NMS ######################

def nms_pytorch(P, scores, thresh_iou):
    areas = torch.prod(P[:, 2:] - P[:, :2], axis=1)   
    order = scores.argsort()
    keep = []
    while len(order) > 0:
        idx = order[-1]
        keep.append(P[idx])
        order = order[:-1]  
        if len(order) == 0:
            break
        
        boxes = torch.index_select(P, dim=0, index=order)
        tl = torch.max(P[idx, :2], boxes[:, :2])
        br = torch.min(P[idx, 2:], boxes[:, 2:]) 
        inter = torch.clamp(torch.prod(br - tl, axis=1), min=0.0)

        areas_other = torch.index_select(areas, dim=0, index=order)   
        union = (areas_other - inter) + areas[idx]           
        IoU = inter / union
        mask = IoU < thresh_iou
        order = order[mask]
    return torch.hstack(keep)

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_cp1 = img.copy()
img_cp2 = img.copy()
img_cp3 = img.copy()

boxes = torch.tensor([
    [139, 200, 207, 301],
    [137, 190, 210, 298],
    [139, 205, 190, 309],
    [145, 210, 215, 280],
    [130, 189, 230, 290]  
])

for i in range(len(boxes)):
    box = boxes[i].numpy()
    img_cp1 = cv2.rectangle(img_cp1, (box[0], box[1]), (box[2], box[3]), (0,255,0), 3)  
    
plt.imshow(img_cp1)
plt.show()
    
scores = torch.tensor([0.99, 0.98, 0.96, 0.97, 0.95])

op = nms_pytorch(boxes, scores, 0.5).numpy()
print(op)

img_cp2 = cv2.rectangle(img_cp2, (op[0], op[1]), (op[2], op[3]), (0,255,0), 3)

plt.imshow(img_cp2)
plt.show()

###################### USING PYTORCH NMS ######################
from torchvision.ops import nms
indices = nms(boxes.float(), scores, 0.5)
op = boxes[indices]
op = op[0].numpy()
print(op)

img_cp3 = cv2.rectangle(img_cp3, (op[0], op[1]), (op[2], op[3]), (0,255,0), 3)

plt.imshow(img_cp3)
plt.show()








