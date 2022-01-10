# train for lines

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

from model_bar import cnet
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import ChartDataset,collate
from loss import FocalLoss, RegL1Loss

root_dir_synth="/content/gdrive/MyDrive/CV Final/ICPR_ChartCompetition2020_AdobeData"
root_dir_pmc = "/content/gdrive/MyDrive/CV Final/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/"
chartype="bar"

def train(root_dir=root_dir_synth,dataset="synth", chart_type=None, img_size=(1024, 1024),path=None,epochs=5):
    model_path=path
    trainset = ChartDataset(root_dir=root_dir, dataset=dataset, chart_type=chartype, img_size=(1024, 1024),
                            heatmap_size=(256, 256))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=False, collate_fn=collate, num_workers=2)
    model=cnet()
    criterion_hm = FocalLoss()
    criterion_wh = nn.MSELoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    loss_meter, it = 0, 0
    for i in range(epochs):
        print("epoch:: ", i)
        for j, data in enumerate(trainloader):
            model.train()
            # if torch.cuda.is_available():
            # [ 0 torch.Tensor(np.array(imgs)),1 torch.Tensor(np.array(hms)),2 pts,3 torch.Tensor(np.array(wh)),4 torch.Tensor(np.array(reg)),5 torch.Tensor(np.array(reg_mask)),6 torch.tensor(np.array(ind)),7 info]
            input_img = data[0].to(device)
            input_hm = data[1].to(device)
            input_wh = data[3].to(device)
            input_reg = data[4].to(device)
            input_regmask = data[5].to(device)
            input_ind = data[6].to(device)
            model.to(device)

            optimizer.zero_grad()

            # print(input_img.shape)
            output_hm, output_hw, output_reg = model(input_img)
            # print(output_hm, output_hw, output_reg)
            hm_loss = criterion_hm(output_hm, input_hm)
            # print(input_wh.shape, output_hw.shape)
            output_hw = output_hw[0].permute(0, 1, 2).view(1 * 256, 256, 2).argmax(axis=1)
            output_reg = output_reg[0].permute(0, 1, 2).view(1 * 256, 256, 2).argmax(axis=1)

            wh_loss = criterion_wh(output_hw, input_wh[0])
            reg_loss = criterion_reg(output_reg, input_reg[0])
            # wh_loss = criterion_wh( output_hw, input_wh, input_regmask)
            # reg_loss = criterion_reg( output_reg, input_reg, input_regmask)
            loss = hm_loss + 0.001 * wh_loss + 0.001 * reg_loss

            print("Loss: ", loss)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            loss_meter += loss.item()

model_save_state = {}
path = model_path
model_save_state = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss}
torch.save(model_save_state, path)

