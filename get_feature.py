
import argparse
import os
import pickle
import random
import shutil
import time
import warnings
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tre_goods import TreGoods

import tensorboard_logger as tb_logger

from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset_path', default='/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all')
# parser.add_argument('--dataset_path', default='/media/ubuntu/lihui/DING/fresh_crop_h_w/4_11_dataset_clean/clean_pkl')
# parser.add_argument('--dataset_path', default='/media/ubuntu/lihui/DING/fresh_crop_h_w/4_15_dataset_trans')
parser.add_argument('--evel_opt', default='10standard')
parser.add_argument('--arch', default='efficientnet-b2')
# parser.add_argument('--weights_path', default='pretrain_pth/21_clean_fc_model_best.pth.tar')
parser.add_argument('--weights_path', default='/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all/21_clean_fc1000_model_best.pth.tar')
parser.add_argument('--feat_save_path', default='get_feature/standard_feat.pkl')
parser.add_argument('--batch_size', default=16, type=int)

def main():
    args = parser.parse_args()

    # create model and load pretrained weight
    model = EfficientNet.from_pretrained(args.arch, advprop=False, num_classes=1000)
    model = nn.DataParallel(model)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weights_path)['state_dict'])
    torch.cuda.set_device('cuda:0')
    model.cuda()

    # create dataloader
    mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
    std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        # lambda x: resize(x),
        transforms.ToTensor(),
        normalize
    ])

    test_loader = DataLoader(TreGoods(data_root=args.dataset_path, partition=args.evel_opt, transform=transform),
                             batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=0)

    feat_dict = get_feature(test_loader, model, args)
    with open(args.feat_save_path, "wb") as fp:  # Pickling
        pickle.dump(feat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def get_feature(test_loader, model, args):
    feat_dict = {}
    for idx, (images, target, _) in enumerate(test_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        _, feat = model(images)

        feature = feat.detach().cpu().numpy()
        count = 0
        for i in target:
            if i not in feat_dict.keys():
                feat_dict[i] = []
            feat_dict[i].append(feature[count])
            count += 1

    return feat_dict

if __name__ == '__main__':
    main()