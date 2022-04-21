"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""
import pickle

import cv2

from train_data import PickleTrainData
import argparse
import os
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
parser.add_argument('--evel_opt', default='val')
parser.add_argument('--arch', default='efficientnet-b2')
parser.add_argument('--weights_path', default='/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all/21_clean_fc1000_model_best.pth.tar')
parser.add_argument('--batch_size', default=256, type=int)

def main():
    args = parser.parse_args()

    # create model
    model = EfficientNet.from_pretrained(args.arch, advprop=False, num_classes=1000)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weights_path)['state_dict'])
    torch.cuda.set_device('cuda:0')
    model.cuda()

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

    pickleTrainData = PickleTrainData(args.dataset_path, (256, 256))
    id_map_class_dict = pickleTrainData.id_class_map
    set_class_lst = range(len(id_map_class_dict))
    # evaluate on validation set
    validate(test_loader, model, id_map_class_dict, set_class_lst, args)

def validate(val_loader, model, id_map_class_dict, set_class_lst, args):
    model.eval()

    imgs = pickle.load(
        open(os.path.join(args.dataset_path, 'treGoods_category_split_train_phase_' + args.evel_opt + '.pickle'),'rb'), encoding='utf-8')
    for set_class in set_class_lst:
        error_ids_lst, error_target_lst = save_errclass_img(val_loader, model, set_class)
        dirs = os.path.join(args.dataset_path, 'error_img', 'class' + str(set_class) + '_' + id_map_class_dict[set_class])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        for i in range(len(error_ids_lst)):
            # imgs = imgs['data'][i]
            idx = int(error_target_lst[i])
            cv2.imwrite(
                os.path.join(dirs, str(error_target_lst[i]) + '_' + str(id_map_class_dict[idx]) + '_' + str(i)) + '.jpg',
                cv2.cvtColor(imgs['data'][error_ids_lst[i]], cv2.COLOR_RGB2BGR))


def save_errclass_img(val_loader, model, set_class):
    """One epoch validation"""
    # switch to evaluate mode
    model.eval()
    error_idxs_lst = []
    error_target_lst = []
    with torch.no_grad():
        for idx, (input, target, _) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            outputs = model(input)
            if type(outputs) == tuple:
                output = outputs[1]
            else:
                output = outputs

            # get error images of class list
            set_class_idx = set_class
            error_idxs, error_target = get_err_img(output, target, set_class_idx)
            for i in range(len(error_idxs)):
                error_idxs[i] = error_idxs[i] + target.size(0) * idx
            error_idxs_lst = error_idxs_lst + error_idxs
            if len(error_target) != 0:
                error_target_lst += error_target

    return error_idxs_lst, error_target_lst

def get_err_img(output, target, set_class):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        error_idxs = []
        error_target = []
        for i in range(len(target)):
            if (target[i] == set_class) & (target[i] != pred[0][i]):
                error_idxs.append(i)
                error_target.append(pred[0][i].cpu().numpy())
        return error_idxs, error_target


if __name__ == '__main__':
    main()
