import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import ChartDataset,collate
import torch.optim as optim
from loss import FocalLoss, RegL1Loss
from train_bar import train
from loss import FocalLoss, RegL1Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "/content"

def test(path):
    root_dir_synth = "/content/gdrive/MyDrive/CV Final/ICPR_ChartCompetition2020_AdobeData"
    root_dir_pmc = "/content/gdrive/MyDrive/CV Final/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/"
    chartype = "box"
    valset = ChartDataset(root_dir=root_dir_synth, dataset="synth", chart_type=chartype, img_size=(1024, 1024),
                            heatmap_size=(256, 256))
    valloader = torch.utils.data.DataLoader(valset, batch_size=3, shuffle=False, collate_fn=collate, num_workers=2)
    model = torch.load(path)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    criterion_hm = FocalLoss()

    model.eval()
    for i in range(1):
        for j, data in enumerate(valloader):
            # if torch.cuda.is_available():
            # img,heatmaps,points,boxes,wh,reg
            input_img = data[0].to(device)
            input_hm = data[1].to(device)
            input_wh = data[3].to(device)
            input_reg = data[4].to(device)
            input_regmask = data[5].to(device)
            input_ind = data[6].to(device)
            model.to(device)
            # print(target.shape)
            output_hm, output_hw, output_reg = model(input_img)
            # print(output_hm, output_hw, output_reg)
            hm_loss = criterion_hm(output_hm, input_hm)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=['train', 'test'], help="train | test")
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number workers in dataloader')
    parser.add_argument('--max_epoch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34, 50, 101, 152],
                        help='resnet numner in [18,34,50,101,152]')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=512, help='image input size')
    parser.add_argument('--max_objs', type=int, default=16, help='max object number in a picture')
    parser.add_argument('--topk', type=int, default=4, help='topk in target')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for nms,default is 0.5')
    parser.add_argument('--down_ratio', type=int, default=4, help='downsample ratio')
    parser.add_argument('--ckpt', type=str, default='w.pth', help='the path of model weight')
    parser.add_argument('--test_img_path', type=str, default='VOC2007/JPEGImages/000019.jpg',
                        help='test image path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    opt = parser.parse_args()

    root_dir_synth = "/content/gdrive/MyDrive/CV Final/ICPR_ChartCompetition2020_AdobeData"
    root_dir_pmc = "/content/gdrive/MyDrive/CV Final/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/"
    chartype = "box"

    train(root_dir=root_dir_synth, dataset="synth", chart_type=chartype, img_size=(1024, 1024),path=path,epochs=5)