"""
Evaluate on treGoods
output a csv file to show error details of each class
"""

import pandas as pd
from train_data import PickleTrainData
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tre_goods import TreGoods
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

    # TODO maybe could be better
    pickleTrainData = PickleTrainData(r'/media/ubuntu/lihui/DING/fresh_crop_h_w', (256, 256))
    id_map_class_dict = pickleTrainData.id_class_map

    # evaluate on validation set
    err_idx, counts, error_class_dict = validate(test_loader, model, id_map_class_dict)

    data_list = []
    row = 0
    for key in error_class_dict.keys():
        temp = [row, key[len(str(row)) + 1:], err_idx[int(row)][0], err_idx[int(row)][1], error_class_dict[key]]
        data_list.append(temp)
        row += 1
    df = pd.DataFrame(data_list)
    df.to_csv(r"temp.csv")


def validate(val_loader, model, id_map_class_dict):

    # switch to evaluate mode
    model.eval()
    set_class = range(239)
    save_class = [5]
    error_targets_lst = []
    error_class_dict = {}
    error_target_lst = {}

    # init
    for i in set_class:
        error_target_lst[i] = [0,0]
        class_name = str(i) + '_' + str(id_map_class_dict[i])
        error_class_dict[class_name] = {}

    counts = 0
    with torch.no_grad():
        for i, (images, target, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(images)
            err_dict, error_img_idxs, error_class, error_target \
                = get_err_dict(output,
                               target,
                               set_class=set_class,
                               save_class=save_class,
                               id_map_class_dict=id_map_class_dict
                               )

            if len(error_target) != 0:
                error_targets_lst += error_target
            for key in err_dict.keys():
                error_target_lst[key][0] += err_dict[key][0]
                error_target_lst[key][1] += err_dict[key][1]

            for key in error_class:
                for pred_key in error_class[key].keys():
                    if pred_key not in error_class_dict[key].keys():
                        error_class_dict[key][pred_key] = 0
                    error_class_dict[key][pred_key] += error_class[key][pred_key]

    return error_target_lst, counts, error_class_dict

def get_err_dict(output, target, set_class, save_class, id_map_class_dict):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        error_dict = {}
        error_img_idxs = []
        error_target = []
        error_class = {}
        for i in set_class:
            error_dict[i] = [0,0]
            class_name = str(i)+'_'+str(id_map_class_dict[i])
            error_class[class_name] = {}
        for i in range(len(target)):
            idx = int(target[i].cpu().numpy())
            error_dict[idx][1] += 1
            if (target[i] != pred[0][i]):
                error_dict[idx][0] += 1
                pred_idx = int(pred[0][i].cpu().numpy())
                class_name_idx = str(idx)+'_'+str(id_map_class_dict[idx])
                pred_class_idx = str(idx)+'_'+str(id_map_class_dict[pred_idx])
                if pred_class_idx not in error_class[class_name_idx].keys():
                    error_class[class_name_idx][pred_class_idx] = 0
                error_class[class_name_idx][pred_class_idx] += 1

                if idx in save_class:
                    error_img_idxs.append(i)
                    error_target.append(pred[0][i].cpu().numpy())
        return error_dict, error_img_idxs, error_class, error_target

if __name__ == '__main__':
    main()
