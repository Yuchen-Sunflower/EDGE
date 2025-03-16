import os
import nni
import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
from utils.anom_utils import ToLabel
from model.classifiersimple import *
from utils.svhn import SVHN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def train(args):
    setup_seed(2024)

    args.save_dir += args.dataset + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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
        loader = pascalVOCLoader(
                                 "./datasets/pascal/", 
                                 img_transform = img_transform, 
                                 label_transform = label_transform)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=img_transform,
                                   label_transform=label_transform)
    elif args.dataset == "coco":
        loader = cocoloader("/data/sunyuchen/datasets/COCO2014_pro/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = cocoloader("/data/sunyuchen/datasets/COCO2014_pro/", split="multi-label-val2014",
                            img_transform = val_transform,
                            label_transform = label_transform)
    elif args.dataset == "nus-wide":
        loader = nuswideloader("./datasets/nus-wide/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                                 img_transform=val_transform,
                                 label_transform=label_transform)
    else:
        raise AssertionError

    # OOD data
    if args.oe_data == "22k":
        oe_root = "/data/sunyuchen/ood_datasets/ImageNet-22K/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "nus_ood":
        oe_root = "/data/sunyuchen/ood_datasets/nus_ood/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "texture":
        oe_root = "/data/sunyuchen/ood_datasets/dtd/images/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform = img_transform)
    elif args.oe_data == "MNIST":
        gray_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize
        ])
        oe_root = "/data/sunyuchen/ood_datasets/MNIST/"
        out_test_data = torchvision.datasets.MNIST(oe_root, train=False, transform=gray_transform, download=True)
    elif args.oe_data == "lsun":
        oe_root = "/data/sunyuchen/ood_datasets/LSUN/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == 'svhn':
        oe_root = "/data/sunyuchen/ood_datasets/svhn/"
        out_test_data = SVHN(oe_root, split='train_and_extra', transform=img_transform)
    elif args.oe_data == "places50":
        oe_root = "/data/sunyuchen/ood_datasets/Places/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "inat":
        oe_root = "/data/sunyuchen/ood_datasets/iNaturalist/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "sun50":
        oe_root = "/data/sunyuchen/ood_datasets/SUN/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    

    args.n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    out_test_loader = data.DataLoader(out_test_data, batch_size=args.oe_batch_size, num_workers=8, shuffle=True, pin_memory=True)
    out_test_loader.dataset.offset = np.random.randint(len(out_test_loader.dataset))

    print("number of images = ", len(loader))
    print("number of classes = ", args.n_classes, " architecture used = ", args.arch)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[:-1])
        clsfier = clssimp(2048, args.n_classes)
        # binary_clsfier = clssimp(2048, 1)
    elif args.arch == "resnet50":
        orig_resnet = torchvision.models.resnet50(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        clsfier = clssimp(num_ftrs, args.n_classes)
        # binary_clsfier = clssimp(num_ftrs, 1)

    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, args.n_classes)
        # binary_clsfier = clssimp(1024, 1)

    device_ids = list(map(int, args.device_id.split(',')))
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print("Available device = ", str(device_ids))
    model.to(device)
    clsfier.to(device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        clsfier = nn.DataParallel(clsfier, device_ids=device_ids)

    S_id_total = torch.zeros((args.batch_size, args.oe_batch_size), dtype=torch.float32)
    S_ood_total = torch.zeros((args.batch_size, args.oe_batch_size), dtype=torch.float32)
    batch_num = 0
    for in_set, out_set in tqdm(zip(trainloader, out_test_loader)):
        batch_num += 1
        inputs = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        inputs, target = Variable(inputs.to(device)), Variable(target.to(device).float())

        # forward
        x = model(inputs)
        feat = x.detach().view(inputs.shape[0], -1)
        f_id = feat[:len(in_set[0])]
        f_ood = feat[len(in_set[0]):]

        U_id, S_id, V_id = torch.svd(f_id)
        U_ood, S_ood, V_ood = torch.svd(f_ood)

        S_id_total += S_id
        S_ood_total += S_ood

        # print(S_id)
        # print(S_ood)
        # print(U_id)
        # print(U_ood)
        # print(torch.mean(torch.pow(S_id[:int(S_id.shape[0] / 3)] - S_ood[:int(S_id.shape[0] / 3)], 2)))
        # print(torch.mean(torch.pow(S_id[int(S_id.shape[0] / 3):int(S_id.shape[0] * 2 / 3)] - S_ood[int(S_id.shape[0] / 3):int(S_id.shape[0] * 2 / 3)], 2)))
        # print(torch.mean(torch.pow(S_id[int(S_id.shape[0] * 2 / 3):] - S_ood[int(S_id.shape[0] * 2 / 3):], 2)))
        # print(torch.mean(torch.pow(S_id - S_ood, 2)))

        # print(torch.norm(S_id[:10] - S_ood[:10]))
        # print(torch.norm(S_id[int(S_id.shape[0] / 3):int(S_id.shape[0] * 2 / 3)] - S_ood[int(S_id.shape[0] / 3):int(S_id.shape[0] * 2 / 3)]))
        # print(torch.norm(S_id[int(S_id.shape[0] * 2 / 3):] - S_ood[int(S_id.shape[0] * 2 / 3):]))
        # print(torch.norm(S_id - S_ood))
        # print(S_id - S_ood)
        # print(feat.shape)
        # print(feat[:len(in_set[0])].shape)
        # print(feat[len(in_set[0]):].shape)
        # in_norm = torch.norm(feat[:len(in_set[0])])
        # out_norm = torch.norm(feat[len(in_set[0]):])
    
    print('S_gap{:.2f}: \t\t\t{:.2f}'.format(args.k, torch.norm(S_id_total[:args.k] / batch_num - S_ood_total[:args.k] / batch_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--oe_data', type=str, default='imagenet')
    parser.add_argument('--n_classes', type=int, default=20, help='# of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--oe_batch_size', type=int, default=64, help='Batch Size')
    # batch_size 320 for resenet101
    parser.add_argument('--k', type=int, default=32, help="top-k singular values")
    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")
    #save and load
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models", help='Path to load models')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    args = parser.parse_args()
    train(args)