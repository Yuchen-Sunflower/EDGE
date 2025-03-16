import os
import nni
import torch
import argparse
import torchvision
import torch.nn as nn
import warnings
import util
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import validate
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
from utils.anom_utils import ToLabel
from model.classifiersimple import *
from torch.optim import lr_scheduler
import lib
from utils import anom_utils
from loss import BCEWithThresholdLoss, AsymmetricLoss, FocalLoss

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

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def train(args):
    setup_seed(2024)
    # alpha = 0.01
    # alpha = params['alpha']
    # args.l_rate = params['l_rate']

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


    args.n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[:-1])
        clsfier = clssimp(2048, args.n_classes)

    elif args.arch == "resnet50":
        orig_resnet = torchvision.models.resnet50(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        clsfier = clssimp(num_ftrs, args.n_classes)
        
    elif args.arch == "resnet34":
        orig_resnet = torchvision.models.resnet34(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        clsfier = clssimp(num_ftrs, args.n_classes)

    elif args.arch == "resnet18":
        orig_resnet = torchvision.models.resnet18(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        clsfier = clssimp(num_ftrs, args.n_classes)

    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, args.n_classes)


    device_ids = list(map(int, args.device_id.split(',')))
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print("Available device = ", str(device_ids))
    model.to(device)
    clsfier.to(device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        clsfier = nn.DataParallel(clsfier, device_ids=device_ids)

    if args.opt == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.l_rate/10},
            {'params': clsfier.parameters()}
        ], lr=args.l_rate)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.parameters(), 'lr': args.l_rate}, 
            {'params': clsfier.parameters(), 'lr': args.l_rate}
        ], weight_decay=args.weight_decay, momentum=0.9, dampening=0, nesterov=True)

    # if args.load:
    #     model.load_state_dict(torch.load(args.save_dir + args.arch + ".pth"))
    #     clsfier.load_state_dict(torch.load(args.save_dir + args.arch +'clsfier' + ".pth"))
    #     print("Model loaded!")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    new_state_dict_cls = OrderedDict()
    state_dict = torch.load(args.save_dir + args.arch + "_joint.pth")
    state_dict_cls = torch.load(args.save_dir + args.arch + '_jointclsfier' + ".pth")
    for k, v in state_dict.items():
        name = "module." + k
        new_state_dict[name] = v
    
    for k, v in state_dict_cls.items():
        name = "module." + k
        new_state_dict_cls[name] = v

    # Load the state dict
    model.load_state_dict(new_state_dict)
    clsfier.load_state_dict(new_state_dict_cls)
    # model.load_state_dict(torch.load(args.save_dir + args.arch + "_joint_bceoe.pth"))
    # clsfier.load_state_dict(torch.load(args.save_dir + args.arch + '_joint_bceoeclsfier' + ".pth"))
    print("Model loaded!")

    if args.score == "vos":
        def log_sum_exp(value, dim=None, keepdim=False):
            """Numerically stable implementation of the operation

            value.exp().sum(dim, keepdim).log()
            """
            import math
            # TODO: torch.max(value, dim=None) threw an error at time of writing
            if dim is not None:
                m, _ = torch.max(value, dim=dim, keepdim=True)
                value0 = value - m
                if keepdim is False:
                    m = m.squeeze(dim)
                return m + torch.log(torch.sum(
                    F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
            else:
                m = torch.max(value)
                sum_exp = torch.sum(torch.exp(value - m))
                # if isinstance(sum_exp, Number):
                #     return m + math.log(sum_exp)
                # else:
                return m + torch.log(sum_exp)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.n_epoch * len(loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.l_rate))
        
        weight_energy = torch.nn.Linear(args.n_classes, 1).to(device)
        torch.nn.init.uniform_(weight_energy.weight)
        data_dict = torch.zeros(args.n_classes, args.sample_number, 128).to(device)
        number_dict = {}
        for i in range(args.n_classes):
            number_dict[i] = 0
        eye_matrix = torch.eye(128).to(device)
        print(eye_matrix.shape)
        logistic_regression = torch.nn.Linear(1, 2)
        logistic_regression = logistic_regression.to(device)

    criterion = nn.BCEWithLogitsLoss()
    criterion_asl = AsymmetricLoss()
    criterion_focal = FocalLoss()
    
    best_mAP = 0
    best_model = None
    for epoch in tqdm(range(args.n_epoch)):
        model.train()
        clsfier.train()
        # feats = torch.zeros((args.batch_size, 2048)).to(device)
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device).float())
                
            optimizer.zero_grad()
            feat = model(images)
            outputs = clsfier(feat)
            if args.score == "normal":
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                # scheduler.step()

            elif args.score == 'asl':

                loss = criterion_asl()
        
        mAP = validate.validate(args, model, clsfier, val_loader, device)
        if best_mAP < mAP:
            best_mAP = mAP
            best_model = model
            if len(device_ids) > 1:
                torch.save(model.module.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.score) + ".pth")
                torch.save(clsfier.module.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.score) + "clsfier.pth")
            else:
                torch.save(model.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.score) + ".pth")
                torch.save(clsfier.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.score) + "clsfier.pth")
            
        print("Epoch [%d/%d] Loss: %.4f mAP: %.4f" % (epoch, args.n_epoch, loss.data, mAP))
    print("Best mAP:", best_mAP)
    # model = best_model

    # OOD data
    if args.ood_data == "imagenet":
        if args.dataset == "nus-wide":
            ood_root = "/data/sunyuchen/ood_datasets/nus_ood/"
            out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
        else:
            ood_root = "/data/sunyuchen/ood_datasets/ImageNet-22K/"
            out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "texture":
        ood_root = "/data/sunyuchen/ood_datasets/dtd/images/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform = img_transform)
    elif args.ood_data == "MNIST":
        gray_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize
        ])
        out_test_data = torchvision.datasets.MNIST('/data/sunyuchen/ood_datasets/MNIST/',
                       train=False, transform=gray_transform, download=True)

    out_test_loader = data.DataLoader(out_test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    in_scores, in_labels = lib.get_logits(val_loader, model, clsfier, args, name="in_test", device=device)
    out_scores, out_labels = lib.get_logits(out_test_loader, model, clsfier, args, name="out_test", device=device)
    
    lib.get_id_auc(in_scores, out_scores, in_labels, args)

    if args.ood == "lof":
        val_scores = lib.get_logits(val_loader, model, clsfier, args, name="in_val", device=device)
        scores = lib.get_localoutlierfactor_scores(val_scores, in_scores, out_scores)
        in_scores = scores[:len(in_scores)]
        out_scores = scores[-len(out_scores):]

    if args.ood == "isol":
        val_scores = lib.get_logits(val_loader, model, clsfier, args, name="in_val", device=device)
        scores = lib.get_isolationforest_scores(val_scores, in_scores, out_scores)
        in_scores = scores[:len(in_scores)]
        out_scores = scores[-len(out_scores):]
    ###################### Measure ######################
    anom_utils.get_and_print_results(in_scores, out_scores, args.ood, args.method)
    print("Best mAP:", best_mAP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--n_epoch', type=int, default=12, help='# of the epochs')
    parser.add_argument('--n_classes', type=int, default=20, help='# of classes')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch Size')
    # batch_size 320 for resenet101
    parser.add_argument('--l_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--score', type=str, default='normal', help='normal|energy|OE|joint')
    parser.add_argument('--opt', type=str, default='adam', help='adam|sgd')

    parser.add_argument('--ood_data', type=str, default='imagenet')
    parser.add_argument('--ood', type=str, default='energy', help='which measure to use odin|M|logit|energy|msp|prob|lof|isol')
    parser.add_argument('--method', type=str, default='sum', help='which method to use max|sum')
    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")

    #save and load
    parser.add_argument('--load', action='store_true', help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models", help='Path to load models')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')

    parser.add_argument('--start_epoch', type=int, default=50, help='# of the epochs')
    parser.add_argument('--sample_number', type=int, default=1000)
    parser.add_argument('--select', type=int, default=1)
    parser.add_argument('--sample_from', type=int, default=10000)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    parser.add_argument('--weight_decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    args = parser.parse_args()

    train(args)