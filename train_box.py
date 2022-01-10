#train for boxes

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms as tf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
import json
import cv2
import torchvision
from torchvision.io import read_image, ImageReadMode
from torch.optim.lr_scheduler import StepLR
import pathlib
from torchsummary import summary
from torchvision.models import resnet
import time

from scipy import ndimage,io
from skimage import io, transform

from model_box import dla34
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import ChartDataset,collate
from loss import FocalLoss, RegL1Loss

root_dir_synth="/content/gdrive/MyDrive/CV Final/ICPR_ChartCompetition2020_AdobeData"
root_dir_pmc = "/content/gdrive/MyDrive/CV Final/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/"
chartype="box"

def train(root_dir=root_dir_synth,dataset="synth", chart_type=None, img_size=(1024, 1024),path=None,epochs=5):
    model_path=path
    trainset = ChartDataset(root_dir=None, dataset=None, chart_type=chartype, img_size=(1024, 1024),
                            heatmap_size=(256, 256))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=False, collate_fn=collate, num_workers=2)
    model=dla34()
    criterion_hm = FocalLoss()
    criterion_wh = RegL1Loss()
    #criterion_wh = nn.MSELoss()
    criterion_reg = RegL1Loss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    loss_meter, it = 0, 0
    for i in range(epochs):
        print("epoch:: ", i)
    for j, data in enumerate(trainloader):
        model.train()
        # if torch.cuda.is_available():
        # img,heatmaps,points,boxes,wh,reg
        input_img = data[0].to(device)
        input_hm = data[1].to(device)
        input_wh = data[3].to(device)
        input_reg = data[4].to(device)
        input_regmask = data[5].to(device)
        input_ind = data[6].to(device)
        model.to(device)

        optimizer.zero_grad()

        # print(target.shape)
        output_hm, output_hw, output_reg = model(input_img)
        # print(output_hm, output_hw, output_reg)
        hm_loss = criterion_hm(output_hm, input_hm)
        # wh_loss = criterion_wh(input_wh, output_hw, input_regmask, input_ind)
        # reg_loss = criterion_reg(input_reg, output_reg,input_regmask, input_ind)
        # loss = my_loss(output, target)

        print("Loss: ", hm_loss)

        hm_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        loss_meter += hm_loss.item()

    model_save_state = {}
    path = model_path
    model_save_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': hm_loss}
    torch.save(model_save_state, path)

