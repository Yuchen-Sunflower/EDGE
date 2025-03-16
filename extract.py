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
from loss import BCEWithThresholdLoss

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
    pert = [-2.1604e+00, -4.5727e+00, -3.5590e+00, -3.9636e+00, -2.5841e-02,
        -2.5234e+00, -1.9096e+00, -3.4089e+00, -2.9760e+00, -2.4546e+00,
        -1.0737e+00, -4.3182e-01, -1.3238e+00, -1.3891e+00, -2.0214e+00,
        -2.1945e+00, -7.2678e-01, -1.2225e+00, -1.0644e+00, -1.3803e+00,
        -1.6024e+00, -1.6286e+00, -1.5711e+00, -1.2797e+00, -1.0831e+00,
        -1.1793e+00, -1.1025e+00, -1.1272e+00, -8.8474e-01, -8.7300e-01,
        -1.0393e+00, -1.1927e+00, -1.3986e+00, -1.3963e+00, -9.7428e-01,
        -1.1678e+00, -9.0830e-01, -9.3781e-01, -8.2658e-01, -5.9882e-01,
        -8.2341e-01, -8.0106e-01, -7.2718e-01, -8.9988e-01, -9.7488e-01,
        -7.9611e-01, -8.7452e-01, -9.0134e-01, -6.5237e-01, -1.0189e+00,
        -9.8208e-01, -7.4341e-01, -6.4656e-01, -6.6394e-01, -8.3135e-01,
        -6.1690e-01, -5.3853e-01, -4.7945e-01, -5.3359e-01, -8.1178e-01,
        -5.4850e-01, -6.9404e-01, -6.2694e-01, -6.4742e-01, -5.5347e-01,
        -4.2297e-01, -2.0710e-01, -1.8511e-01, -9.1071e-02, -2.0046e-01,
        -2.1980e-01, -2.9787e-01, -2.1161e-01, -2.8125e-01, -4.6269e-01,
        -3.5546e-01, -2.9370e-01, -1.4454e-01, -2.2420e-01, -1.9376e-01,
        -1.5308e-01, -1.7348e-01, -1.7100e-01, -1.4743e-01, -1.4823e-01,
        -1.8924e-01, -2.5414e-01, -2.6536e-01, -1.7812e-01, -2.0831e-01,
        -2.3280e-01, -2.2755e-01, -3.4174e-01, -1.7587e-01, -1.8416e-01,
        -2.8174e-01, -2.4393e-01, -1.8724e-01, -1.1854e-01, -1.1955e-01,
        -9.2806e-02, -9.9387e-02, -7.9635e-02, -6.2731e-02, -1.2882e-01,
        -1.3082e-01, -2.3599e-01, -2.0358e-01, -1.8737e-01, -2.3450e-01,
        -1.9581e-01, -2.7854e-02, -1.9501e-02,  1.2077e-02, -9.3476e-02,
        -3.3150e-03, -1.6759e-01, -1.3381e-01, -1.2812e-01, -1.4297e-01,
        -1.1769e-01, -1.8572e-01,  2.8805e-02, -1.2283e-01, -6.5096e-02,
        -7.4374e-02, -1.2407e-01, -7.5517e-02]
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
        ], weight_decay=1e-4, momentum=0.9, dampening=0, nesterov=True)

    # if args.load:
    #     model.load_state_dict(torch.load(args.save_dir + args.arch + ".pth"))
    #     clsfier.load_state_dict(torch.load(args.save_dir + args.arch +'clsfier' + ".pth"))
    #     print("Model loaded!")
    # if args.load:
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # new_state_dict_cls = OrderedDict()
    # state_dict = torch.load(args.save_dir + args.arch + "_joint_bceoe150.pth")
    # state_dict_cls = torch.load(args.save_dir + args.arch + '_joint_bceoe150clsfier' + ".pth")
    # for k, v in state_dict.items():
    #     name = "module." + k
    #     new_state_dict[name] = v
    
    # for k, v in state_dict_cls.items():
    #     name = "module." + k
    #     new_state_dict_cls[name] = v

    # # Load the state dict
    # model.load_state_dict(new_state_dict)
    # clsfier.load_state_dict(new_state_dict_cls)
    # # model.load_state_dict(torch.load(args.save_dir + args.arch + "_joint_bceoe.pth"))
    # # clsfier.load_state_dict(torch.load(args.save_dir + args.arch + '_joint_bceoeclsfier' + ".pth"))
    # print("Model loaded!")

    criterion = nn.BCEWithLogitsLoss()
    steps_per_epoch = len(trainloader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.l_rate, steps_per_epoch=steps_per_epoch, epochs=args.n_epoch, pct_start=0.2)
    
    best_mAP = 0
    best_model = None
    for epoch in tqdm(range(args.n_epoch)):
        model.train()
        clsfier.train()
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device).float())
                
            optimizer.zero_grad()
         
            feat = model(images)
            outputs_id = clsfier(feat)
            loss = criterion(outputs_id, labels)
            # print(loss)

            U_id, S_id, V_id = torch.svd(feat.detach().view(feat.shape[0], -1))
            sing_num = len(S_id)
            # pert = torch.randn(sing_num).to(device)
            pert = torch.tensor(pert).to(device)
            S_ood = pert[:feat.shape[0]] + S_id
            S_diag_ood = torch.diag(S_ood)

            # print(U_id.shape)
            # print(S_diag_ood.shape)
            # print(V_id.shape)
            feat_ood = (U_id @ S_diag_ood @ V_id.t()).reshape(feat.shape)
            # print(feat_ood.shape)
            outputs_ood = clsfier(feat_ood)

            Ec_in = torch.log(1 + torch.exp(outputs_id)).sum(1)
            Ec_out = torch.log(1 + torch.exp(outputs_ood)).sum(1)
            # Ec_in_bottom_k, _ = torch.topk(-Ec_in, args.k)
            # Ec_in_bottom_k = -Ec_in_bottom_k

            # in_energy_loss = F.relu(args.m_in - Ec_in_bottom_k).mean()
            # out_energy_loss = F.relu(Ec_out - args.m_out).mean()
            # loss_pair = args.energy_beta * (out_energy_loss + in_energy_loss)

            in_energy_loss = F.relu(args.m_in - Ec_in).mean()
            out_energy_loss = F.relu(Ec_out - args.m_out).mean()
            loss_pair = out_energy_loss + in_energy_loss
            # loss_oe = -torch.log(1 - torch.sigmoid(outputs_ood)).mean()

            # loss = loss + args.alpha * loss_oe + energy_beta * loss_pair 
            loss = loss + args.energy_beta * loss_pair 
            # loss = loss + loss_oe
            # print(loss_pair)
            # print(torch.norm(S_id - S_ood))
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        mAP = validate.validate(args, model, clsfier, val_loader, device)
        # nni.report_intermediate_result(mAP)
        if best_mAP < mAP:
            best_mAP = mAP
            best_model = model
            if len(device_ids) > 1:
                torch.save(model.module.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + ".pth")
                torch.save(clsfier.module.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "clsfier.pth")
            else:
                torch.save(model.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + ".pth")
                torch.save(clsfier.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "clsfier.pth")
            
        print("Epoch [%d/%d] Loss: %.4f mAP: %.4f" % (epoch, args.n_epoch, loss.data, mAP))
    print("Best mAP:", best_mAP)
    # nni.report_final_result(best_mAP)
    model = best_model

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
    parser.add_argument('--opt', type=str, default='adam', help='adam|sgd')

    parser.add_argument('--ood_data', type=str, default='imagenet')
    parser.add_argument('--ood', type=str, default='energy', help='which measure to use odin|M|logit|energy|msp|prob|lof|isol')
    parser.add_argument('--m_in', type=float, default=25., help='default: 25. margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=7., help='default: 7. margin for out-distribution; below this value will be penalized')
    parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
    parser.add_argument('--score', default='joint', type=str, help='fine tuning mode')
    parser.add_argument('--method', type=str, default='sum', help='which method to use max|sum')
    parser.add_argument('--k', type=int, default=50, help='bottom-k for ID')
    parser.add_argument('--alpha', default=1, type=float, help='alpha for ap loss')
    parser.add_argument('--m', default=1, type=float, help='gap between id and ood')
    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")

    #save and load
    parser.add_argument('--load', action='store_true', help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models", help='Path to load models')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    args = parser.parse_args()
    # params = nni.get_next_parameter()
    train(args)