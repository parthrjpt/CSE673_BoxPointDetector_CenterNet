from google.colab import drive
import glob
import json
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary
from torchvision.models import resnet
import torchvision
import os
import glob
import json
import time

from scipy import ndimage,io
from skimage import io, transform



##### NEED TO SET Paths before usage ####
'''
root_dir_synth = "/content/gdrive/MyDrive/CV Final/ICPR_ChartCompetition2020_AdobeData"
#root_dir_pmc = "/content/gdrive/MyDrive/CV Final/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21"

chart_type = "line"
trainset = ChartDataset(root_dir=root_dir_synth,dataset="synth",chart_type=chart_type,img_size=(512,512),heatmap_size=(128,128))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate, num_workers=1)
# Testing the loader
batch = next(iter(trainloader))
'''

data_directory='/content'
dataset_directory=data_directory+'/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21'
target_list=[ 'area', 'heatmap','horizontal_interval', 'manhattan', 'map', 'pie', 'scatter-line','surface', 'venn', 'vertical_box','vertical_interval', 'vertical_bar', 'scatter','line']
annotations_directory=dataset_directory+'/annotations_JSON'
images_directory=dataset_directory+'/images'
chart_dict = {
  "hbox":"box",
  "hGroup":"bar",
  "hStack":"bar",
  "vbox":"box",
  "vGroup":"bar",
  "vStack":"bar",
  "line":"line",
  "scatter":"scatter"
}

chart_dict_synth = {
    "box":["hbox","vbox"],
    "bar":["hGroup","hStack","vGroup","vStack"],
    "line":["line"],
    "scatter":["scatter"]
}

chart_dict_pmc = {
    "box":["vertical_box"],
    "bar":["vertical_bar"],
    "line":["line"],
    "scatter":["scatter"]
}

def cleanData(root_dir):
  for chart_type in ["horizontal_bar","vertical_bar","vertical_box","line","scatter"]:
    image_dir = os.path.join(root_dir,"images",chart_type)
    annot_dir = os.path.join(root_dir,"annotations_JSON",chart_type)
    image_filenames = os.listdir(annot_dir)
    print(len(image_filenames))
    continue
    good_files = []
    for i,filename in enumerate(image_filenames):
      if filename[-5:] != ".json" and filename[-5:] != ".JSON":
        good_files.append(False)
        continue
      with open(os.path.join(annot_dir,filename)) as annot_file:
        lines = annot_file.readlines()
        if len(lines) == 0:
          good_files.append(False)
          continue
        try:
          good_files.append(json.loads("".join(lines))["task6"]!=None)
        except KeyError:
          good_files.append(False)
    for i,filename in enumerate(image_filenames):
      print(os.path.join(annot_dir,filename))
      if not good_files[i]:
        with open(os.path.join(annot_dir,filename)) as annot_file:
          print(annot_file.readlines())
        os.remove(os.path.join(annot_dir,filename))


def normalize(img):
    mx = np.max(img)
    mn = np.min(img)
    return (img - mn) / (mx - mn)


class ChartDataset(Dataset):
    def __init__(self, root_dir, dataset="synth", chart_type="line", img_size=(1024, 1024), heatmap_size=(256, 256),
                 save_heatmaps=False,train=True):
        if dataset == "synth":
            chart_dict = chart_dict_synth
            self.image_dir = os.path.join(root_dir, "ICPR", "Charts")
            self.annot_dir = os.path.join(root_dir, "JSONs")
        elif dataset == "pmc":
            chart_dict = chart_dict_pmc
            self.image_dir = os.path.join(root_dir, "images")
            self.annot_dir = os.path.join(root_dir, "annotations_JSON")
        else:
            print('Invalid dataset, try "synth" or "pmc".')
            return
        if chart_type not in chart_dict.keys():
            print('Incorrect chart type, try "box", "bar", "line", or "scatter".')
            return
        self.chart_type = chart_type

        self.image_filenames = []
        for chart_type in chart_dict[self.chart_type]:
            image_filenames = os.listdir(os.path.join(self.annot_dir, chart_type))
            self.image_filenames.extend([(chart_type, fn[:-5]) for fn in image_filenames])
        if train == True:
            self.image_filenames = self.image_filenames[0: split - 1]
        else:
            self.image_filenames = self.image_filenames[split - 1:]
        self.dataset = dataset
        self.chart_dict = chart_dict
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.heatmap_dir = os.path.join(root_dir, "heatmaps")
        self.save_heatmaps = save_heatmaps

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        info = {}
        t0 = time.time()
        img_type, img_filename = self.image_filenames[idx]

        wh = np.zeros((256, 2), dtype=np.float32)
        # regression
        reg = np.zeros((256, 2), dtype=np.float32)
        # index in 1D heatmap
        ind = np.zeros(256, dtype=np.int)
        # 1=there is a target in the list 0=there is not
        reg_mask = np.zeros(256, dtype=np.uint8)

        if self.dataset == "synth":
            img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_type, img_filename + ".png")),
                               cv2.COLOR_BGR2RGB)
            # print(os.path.join(self.image_dir,img_type,img_filename+".png"))
            info['path'] = os.path.join(self.image_dir, img_type, img_filename + ".png")
        else:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_type, img_filename + ".jpg")),
                               cv2.COLOR_BGR2RGB)
            info['path'] = os.path.join(self.image_dir, img_type, img_filename + ".jpg")
        real_h, real_w, _ = img.shape
        # plt.imshow(img)
        # plt.figure()
        # print('Original Image Size:',real_w,real_h)
        w_ratio = self.img_size[1] / real_w / 4
        h_ratio = self.img_size[0] / real_h / 4
        info['scale_factors'] = [w_ratio, h_ratio]
        # print(w_ratio, h_ratio)
        t1 = time.time()
        # print()
        with open(os.path.join(self.annot_dir, img_type, img_filename + ".json")) as annot_file:
            annots = json.loads("".join(annot_file.readlines()))["task6"]

        info['task2_output'] = (annots['input']['task2_output'])
        info['task3_output'] = (annots['input']['task3_output'])
        info['task4_output'] = (annots['input']['task4_output'])
        info['task5_output'] = (annots['input']['task5_output'])

        # print(annots['input']['task3_output'])
        t2 = time.time()
        # print(os.path.join(self.annot_dir,img_type,img_filename+".json"))
        if os.path.exists(os.path.join(self.heatmap_dir, img_type, img_filename + ".npz")):
            heatmaps = np.stack(np.load(os.path.join(self.heatmap_dir, img_type, img_filename + ".npz")))
            create_heatmaps = False
        elif os.path.exists(os.path.join(self.heatmap_dir, img_type, img_filename + ".npy")):
            heatmaps = np.load(os.path.join(self.heatmap_dir, img_type, img_filename + ".npy"))
            create_heatmaps = False
        else:
            create_heatmaps = True
        scaling_factors = np.divide(trainset.heatmap_size, img.shape[:2])
        t3 = time.time()
        points = []
        boxes = [[], [], []]

        if self.chart_type == "bar":
            # print('in')

            heatmaps = np.zeros((1, self.heatmap_size[0], self.heatmap_size[1]))
            # print('Number of Bars:',np.asarray(annots["output"]["visual elements"]["bars"]).shape)
            for i, bar in enumerate(annots["output"]["visual elements"]["bars"]):
                h, w, x0, y0 = bar.values()
                # print('Before:',h,w,x0,y0)
                h, w, x0, y0 = h * h_ratio, w * w_ratio, x0 * w_ratio, y0 * h_ratio
                # print('After',h,w,x0,y0)
                center = np.array([(x0 + (w / 2)), (y0 + (h / 2))], dtype=np.float32)
                center_int = center.astype(np.int)
                reg[i] = center - center_int
                wh[i] = 1. * w, 1. * h
                ind[i] = center_int[1] * self.heatmap_size[0] + center_int[0]
                # print(center_int)
                # print(0,center_int[0],center_int[1])
                heatmaps[0, center_int[1], center_int[0]] = 1
                # break
                boxes[0].append((x0, y0))
                boxes[1].append((x0 + w, y0 + h))
                boxes[2].append((x0 + w // 2, y0 + h // 2))

                reg_mask[i] = 1
            # boxes = [corner_pts*np.flipud(scaling_factors) for corner_pts in boxes]
        elif self.chart_type == "box":
            # print(img.shape[:2])
            if create_heatmaps:
                heatmaps = np.zeros((5, self.heatmap_size[0], self.heatmap_size[1]))
                # print(heatmaps.shape)
            for i, bbox in enumerate(annots["output"]["visual elements"]["boxplots"]):
                for k, j in enumerate(bbox):
                    box = bbox[j]['_bb']
                    # print(j, box)
                    h, w, x0, y0 = box.values()
                    # print('Before:',h,w,x0,y0)
                    h, w, x0, y0 = h * h_ratio, w * w_ratio, x0 * w_ratio, y0 * h_ratio
                    # print('After',h,w,x0,y0)
                    center = np.array([(x0 + (w / 2)), (y0 + (h / 2))], dtype=np.float32)
                    center_int = center.astype(np.int)
                    reg[i * 5 + k] = center - center_int
                    wh[i * 5 + k] = 1. * w, 1. * h
                    reg_mask[i * 5 + k] = 1
                    ind[i * 5 + k] = center_int[1] * self.heatmap_size[0] + center_int[0]
                    # print(center_int)
                    # print(0,center_int[0],center_int[1])
                    heatmaps[k, center_int[1], center_int[0]] = 1
                    boxes[0].append((x0, y0))
                    boxes[1].append((w, h))
                    boxes[2].append(((x0 + w) // 2, (y0 + h) // 2))
                points.append([])
            # print(c)
            # ind.append(center_int[1] * (256) + center_int[0])
            #   for i,part in enumerate(box.values()):
            #     x,y = part["x"],part["y"]
            #     x,y = int(np.clip(x,0,heatmaps.shape[2]-1)),int(np.clip(y,0,heatmaps.shape[1]-1))
            #     if create_heatmaps:
            #       heatmaps[3,y,x] = 1
            #     points[-1].append((x,y))
            # points = [pts*np.flipud(scaling_factors) for pts in points]
            # boxes = [corner_pts*np.flipud(scaling_factors) for corner_pts in boxes]
        elif self.chart_type == "line":
            heatmaps = np.zeros((1, self.heatmap_size[0], self.heatmap_size[1]))
            # print(heatmaps.shape)
            c = 0
            for i, line in enumerate(annots["output"]["visual elements"]["lines"]):
                points.append([])
                for point in line:
                    h, w = 0, 0
                    x, y = point.values()

                    x, y = int(x), int(y)
                    # print(x,y)
                    h, w, x, y = h * h_ratio, w * w_ratio, x * w_ratio, y * h_ratio
                    center = np.array([(x + (w / 2)), (y + (h / 2))], dtype=np.float32)
                    center_int = center.astype(np.int)
                    reg[c] = center - center_int
                    wh[c] = 1. * w, 1. * h
                    ind[c] = center_int[1] * self.heatmap_size[0] + center_int[0]
                    reg_mask[c] = 1
                    c = c + 1
                    heatmaps[0, center_int[1], center_int[0]] = 1
                    points[-1].append((x, y))
            points = [pts * np.array([[1, 1]]) for pts in points]
        elif self.chart_type == "scatter":
            if self.dataset == "synth":
                scatter_points = [annots["output"]["visual elements"]["scatter points"]]
            else:
                scatter_points = annots["output"]["visual elements"]["scatter points"]

            heatmaps = np.zeros((1, self.heatmap_size[0], self.heatmap_size[1]))
            c = 0
            for i, line in enumerate(scatter_points):
                # print(line)
                for point in (line):
                    h, w = 0, 0
                    x, y = point.values()

                    x, y = int(x), int(y)
                    h, w, x, y = h * h_ratio, w * w_ratio, x * w_ratio, y * h_ratio
                    center = np.array([(x + (w / 2)), (y + (h / 2))], dtype=np.float32)
                    center_int = center.astype(np.int)
                    reg[c] = center - center_int
                    wh[c] = 1. * w, 1. * h
                    ind[c] = center_int[1] * self.heatmap_size[0] + center_int[0]
                    reg_mask[c] = 1
                    c = c + 1
                    # print(center_int)
                    # print(center_int)
                    # print(0,center_int[0],center_int[1])
                    heatmaps[0, center_int[1], center_int[0]] = 1
        t4 = time.time()
        # print(c)
        c = 0
        if create_heatmaps:
            heatmaps = normalize(ndimage.filters.gaussian_filter(heatmaps, [0, *([0.6, 0.6] / scaling_factors)]))
        t5 = time.time()
        img = transform.resize(np.transpose(img, (2, 0, 1)), (3, *self.img_size))
        # if create_heatmaps:
        #   heatmaps = transform.resize(heatmaps,(heatmaps.shape[0],*self.heatmap_size))
        if self.save_heatmaps and create_heatmaps:
            if not os.path.exists(os.path.join(self.heatmap_dir, img_type)):
                os.makedirs(os.path.join(self.heatmap_dir, img_type))
            np.savez(os.path.join(self.heatmap_dir, img_type, img_filename + ".npy"), *[hm for hm in heatmaps])
            # np.save(os.path.join(self.heatmap_dir,img_type,img_filename+".npy"),heatmaps)
        elif self.save_heatmaps and not os.path.exists(os.path.join(self.heatmap_dir, img_type, img_filename + ".npz")):
            np.savez(os.path.join(self.heatmap_dir, img_type, img_filename + ".npz"), *[hm for hm in heatmaps])
        t6 = time.time()
        times = np.array([t0, t1, t2, t3, t4, t5, t6])
        # print(times-np.roll(times,1))
        # print(img.shape,heatmaps.shape,len(points),len(boxes),'WH:',wh.shape,'REG:', reg.shape,ind.shape)
        # print(torch.tensor(boxes))
        return img, heatmaps, points, boxes, wh, reg, reg_mask, ind, info

def collate(batch):
  # print(batch)
  imgs = [sample[0] for sample in batch]
  # print(imgs[0].shape)
  hms = [sample[1] for sample in batch]
  # print(hm.shape)
  pts = [sample[2] for sample in batch]
  # print(pts[0])
  bxs = [sample[3] for sample in batch]
  # print(type(bxs[0][0]))
  wh = [sample[4] for sample in batch]
  # print(wh[0])
  reg = [sample[5] for sample in batch]
  # print(reg[0].shape)
  reg_mask= [sample[6] for sample in batch]
  # print(len(reg_mask))
  ind= [sample[7] for sample in batch]
  # print(len(ind))
  info=[sample[8] for sample in batch]
  return [torch.Tensor(np.array(imgs)),torch.Tensor(np.array(hms)),pts,torch.Tensor(np.array(wh)),torch.Tensor(np.array(reg)),torch.Tensor(np.array(reg_mask)),torch.tensor(np.array(ind)), info]



