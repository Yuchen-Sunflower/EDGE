import os
import nni
import torch
import argparse
import torchvision
import torch.nn as nn
import warnings
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
import lib
from utils import anom_utils
import umap
import matplotlib.pyplot as plt
import pandas as pd

from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
from utils.anom_utils import ToLabel
from model.classifiersimple import *
from torch.optim import lr_scheduler
from torchvision.models import resnet50
from utils.svhn import SVHN

def extract_features(dataset, model):
    model.eval()
    features_list = []
    with torch.no_grad():
        for data, _ in tqdm(dataset):
            data = data.cuda()
            features = model(data).view(data.shape[0], -1)
            # _, features = model.forward_virtual(data)
            features_list.append(features.detach().cpu())
    return torch.vstack(features_list)


def main(args):
    reducer = umap.UMAP()
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    img_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.5, 2.0)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    label_transform = torchvision.transforms.Compose([
            ToLabel(),
        ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    if args.dataset == "pascal":
        train_data = pascalVOCLoader(
                                    "./datasets/pascal/", 
                                    img_transform = img_transform, 
                                    label_transform = label_transform)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                    img_transform=img_transform,
                                    label_transform=label_transform)
    elif args.dataset == "coco":
        train_data = cocoloader("/data/sunyuchen/datasets/COCO2014_pro/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = cocoloader("/data/sunyuchen/datasets/COCO2014_pro/", split="multi-label-val2014",
                            img_transform = val_transform,
                            label_transform = label_transform)
    elif args.dataset == "nus-wide":
        train_data = nuswideloader("./datasets/nus-wide/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                                    img_transform=val_transform,
                                    label_transform=label_transform)

    gray_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize
    ])

    trainloader = data.DataLoader(train_data, batch_size=200, num_workers=8, shuffle=True, pin_memory=True)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[:-1])

    elif args.arch == "resnet50":
        orig_resnet = torchvision.models.resnet50(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])

    elif args.arch == "resnet34":
        orig_resnet = torchvision.models.resnet34(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])

    elif args.arch == "resnet18":
        orig_resnet = torchvision.models.resnet18(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])

    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))

    print(f"./saved_models/{args.dataset}/{args.arch}_{args.score}.pth")
    model.cuda()
    model.load_state_dict(torch.load(f"./saved_models/{args.dataset}/{args.arch}_{args.score}.pth"))
    nuswide_features = extract_features(trainloader, model)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(nuswide_features)
    df = pd.DataFrame(embedding, columns = ['x', 'y'])
    df['label'] = [args.dataset if i < len(nuswide_features) else args.oe_data for i in range(len(embedding))]
    df.to_csv(f'{args.dataset}_{args.score}_embedding.csv', index=False)
    print(embedding)

    # 可视化
    plt.scatter(embedding[:, 0], embedding[:, 1], label=args.dataset, s=2)
    plt.legend(fontsize=18, loc="upper right", markerscale=8)
    plt.xticks([])  # 移除x轴刻度
    plt.yticks([])  # 移除y轴刻度
    plt.savefig(f'./umap/{args.dataset}_{args.score}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)  # 保存为PNG格式，分辨率为300dpi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')

    parser.add_argument('--n_classes', type=int, default=20, help='# of classes')
    # batch_size 320 for resenet101
    parser.add_argument('--l_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--loss', type=str, default='bce', help='bce|softmax|LDAM')
    parser.add_argument('--opt', type=str, default='adam', help='adam|sgd')

    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")
    parser.add_argument('--m_in', type=float, default=-25., help='default: -25. margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='default: -7. margin for out-distribution; below this value will be penalized')
    parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
    parser.add_argument('--score', default='joint', type=str, help='fine tuning mode')
    parser.add_argument('--ood', type=str, default='energy', help='which measure to use odin|M|logit|energy|msp|prob|lof|isol')
    parser.add_argument('--method', type=str, default='sum', help='which method to use max|sum')
    # parser.add_argument('--k', type=int, default=50, help='bottom-k for ID')
    # parser.add_argument('--alpha', default=1, type=float, help='alpha for conf loss')
    # parser.add_argument('--m', default=1, type=float, help='gap between id and ood')

    #save and load
    parser.add_argument('--load', type=bool, default=False, help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models", help='Path to load models')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    args = parser.parse_args()
    # params = nni.get_next_parameter()
    main(args)