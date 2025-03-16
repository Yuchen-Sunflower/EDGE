import os
import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
import lib
from utils import anom_utils

import validate
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
from utils.anom_utils import ToLabel
from model.classifiersimple import *
from torch.optim import lr_scheduler
from utils.svhn import SVHN
from loss import LogitNormLoss, AsymmetricLoss, FocalLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

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
    if args.oe_data == "imagenet":
        if args.dataset == "nus-wide":
            oe_root = "/data/sunyuchen/ood_datasets/nus_ood/"
            out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
        else:
            oe_root = "/data/sunyuchen/ood_datasets/ImageNet-22K/"
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
        # out_test_data = SVHN('/data/sunyuchen/ood_datasets/nus_ood/svhn/', split='train_and_extra',
        #                   transform=torchvision.transforms.ToTensor(), download=False)
        out_test_data = SVHN(oe_root, split='test', transform = torchvision.transforms.ToTensor())
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

    label_npy_file = args.save_dir + args.dataset + "_label_sum.npy"
    print(label_npy_file, os.path.exists(label_npy_file))
    cls_num_list = torch.zeros((1, args.n_classes), dtype=torch.int)
    if not os.path.exists(label_npy_file):
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(trainloader)):
                labels = labels.float()
                cls_num_list = cls_num_list + torch.sum(labels, dim=0)

        cls_num_list = cls_num_list.numpy()
        os.makedirs(args.save_dir, exist_ok = True)
        np.save(args.save_dir + args.dataset + "_label_sum", cls_num_list)
        cls_num_list = cls_num_list[0].tolist()

    else:
        cls_num_list = np.load(label_npy_file)[0].tolist()

    print("Instance number per class = ", cls_num_list)
    print("number of images = ", len(loader))
    print("number of classes = ", args.n_classes, " architecture used = ", args.arch)
    base_probs = []
    for i in range(len(cls_num_list)):
        base_probs.append(cls_num_list[i] / len(loader))

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
        # binary_clsfier = clssimp(1024, 1)

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
            {'params': model.parameters(), 'lr': args.l_rate},
            {'params': clsfier.parameters()}
        ], lr=args.l_rate)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.parameters(), 'lr': args.l_rate}, 
            {'params': clsfier.parameters(), 'lr': args.l_rate}
        ], weight_decay=1e-4, momentum=0.9, dampening=0, nesterov=True)

    if args.score == 'joint':
        steps_per_epoch = len(trainloader)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.l_rate, steps_per_epoch=steps_per_epoch, epochs=args.n_epoch, pct_start=0.2)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.n_epoch * len(trainloader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.l_rate))

    base_probs_tensor = torch.tensor(base_probs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # criterion_asl = AsymmetricLoss()
    # criterion_focal = FocalLoss()
    
    best_mAP = 0
    best_model = None
    energy_beta = 0
    mean_energy = 0
    print(args.alpha)
    for epoch in tqdm(range(args.n_epoch)):
        model.train()
        clsfier.train()
        if epoch >= args.start_epoch:
            energy_beta = args.energy_beta
            # args.l_rate *= 10
        # for i, (images, labels) in tqdm(enumerate(trainloader)):
        for in_set, out_set in tqdm(zip(trainloader, out_test_loader)):
            inputs = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]
            inputs, target = Variable(inputs.to(device)), Variable(target.to(device).float())

            # forward
            x = model(inputs)
            cat_output = clsfier(x)
            output_id = clsfier(x[:len(in_set[0])])

            # backward
            optimizer.zero_grad()
            
            if args.score == 'energy':
                loss = F.cross_entropy(cat_output[:len(in_set[0])], target)
                Ec_out = -torch.logsumexp(cat_output[len(in_set[0]):], dim=1)
                Ec_in = -torch.logsumexp(cat_output[:len(in_set[0])], dim=1)
                loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
                # print('energy')
            elif args.score == 'OE':
                loss = F.cross_entropy(cat_output[:len(in_set[0])], target)
                loss += 0.5 * -(cat_output[len(in_set[0]):].mean(1) - torch.logsumexp(cat_output[len(in_set[0]):], dim=1)).mean()
                # print('OE')
            elif args.score == 'energyML':
                loss = criterion(cat_output[:len(in_set[0])], target)
                Ec_out = -torch.logsumexp(cat_output[len(in_set[0]):], dim=1)
                Ec_in = -torch.logsumexp(cat_output[:len(in_set[0])], dim=1)
                loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
                # print('energy')
            elif args.score == 'OEML':
                loss = criterion(cat_output[:len(in_set[0])], target)
                loss += 0.5 * -(cat_output[len(in_set[0]):].mean(1) - torch.logsumexp(cat_output[len(in_set[0]):], dim=1)).mean()
                # print('OE')
            elif args.score == "OECC":
                loss = F.cross_entropy(cat_output[:len(in_set[0])], target)
                ## OECC Loss Function
                if args.dataset == 'pascal':
                    A_tr = 0.8801  # mAP of pascal baseline model
                elif args.dataset == 'coco':
                    A_tr = 0.7573  # mAP of coco baseline model   
                elif args.dataset == 'nus-wide':
                    A_tr = 0.6018  # mAP of nus-wide baseline model  
                sm = torch.nn.Softmax(dim=1) # Create a Softmax 
                probabilities = sm(cat_output) # Get the probabilites for both In and Outlier Images
                max_probs, _ = torch.max(probabilities, dim=1) # Take the maximum probabilities produced by softmax
                prob_diff_in = max_probs[:len(in_set[0])] - A_tr # Push towards the training accuracy of the baseline
                loss += args.lambda_1 * torch.sum(prob_diff_in**2) ## 1st Regularization term
                prob_diff_out = probabilities[len(in_set[0]):][:] - (1 / args.n_classes)
                loss += args.lambda_2 * torch.sum(torch.abs(prob_diff_out)) ## 2nd Regularization term

            elif args.score == "ATOM":
                in_len = len(in_set[0])
                out_len = len(out_set[0])

                in_target = target
                out_input = inputs[len(in_set[0]):]
                out_target = out_set[1]
                out_target = Variable(out_target.to(device).float())

                attack_out = lib.LinfPGDAttack(model=model, eps=8, nb_iter=5, eps_iter=2, targeted=False, rand_init=True, num_classes=args.n_classes+1, loss_func='CE', elementwise_best=True)
                adv_out_input = attack_out.perturb(out_input[int(out_len/2):], out_target[int(out_len/2):])

                cat_input = torch.cat((inputs, out_input[:int(out_len/2)], adv_out_input), 0)
                x = model(cat_input)
                cat_output = clsfier(x)

                in_output = cat_output[:len(in_set[0])]
                in_loss = F.cross_entropy(in_output, in_target)

                out_output = cat_output[len(in_set[0]):]
                out_loss = F.cross_entropy(out_output, out_target)

                loss = in_loss + args.energy_beta * out_loss

            elif args.score == "SOFL":
                
                num_reject_classes = 10
                in_len = len(in_set[0])
                out_len = len(out_set[0])
                # x = model(inputs)
                # cat_output = clsfier(x)

                nat_in_output = cat_output[:in_len]
                nat_in_loss = F.cross_entropy(nat_in_output, target)

                nat_out_output = cat_output[in_len:]
                pseudo_labels = torch.argmax(nat_out_output, dim=1)
                random_labels = torch.LongTensor(out_len).random_(args.n_classes, args.n_classes + num_reject_classes).cuda()
                out_target = torch.where(pseudo_labels < args.n_classes, random_labels, pseudo_labels)
                nat_out_loss = F.cross_entropy(nat_out_output, out_target)

                # compute gradient and do SGD step
                loss = nat_in_loss + args.beta * nat_out_loss

            elif args.score == 'joint':

                loss = criterion(cat_output[:len(in_set[0])], target)
                E = torch.log(1 + torch.exp(cat_output)).sum(1)
                Ec_in = E[:len(in_set[0])]
                Ec_out = E[len(in_set[0]):]
                Ec_in_bottom_k, _ = torch.topk(-Ec_in, args.k)
                Ec_in_bottom_k = -Ec_in_bottom_k
                Ec_out_max = torch.max(Ec_out)
                Ec_out_max = torch.full_like(Ec_in, Ec_out_max.item())

                loss_pair = F.relu(Ec_out.unsqueeze(1) - Ec_in_bottom_k.unsqueeze(0) + args.m).mean()  
                loss_oe = -torch.log(1 - torch.sigmoid(cat_output[len(in_set[0]):])).mean()

                loss = loss + args.alpha * loss_oe + energy_beta * loss_pair 

            loss.backward()
            optimizer.step()
            scheduler.step()

        mAP = validate.validate(args, model, clsfier, val_loader, device)
        if best_mAP < mAP:
            best_mAP = mAP
            best_model = model
            if len(device_ids) > 1:
                torch.save(model.module.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.alpha) + "_" + str(args.energy_beta) + ".pth")
                torch.save(clsfier.module.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.alpha) + "_" + str(args.energy_beta) + "clsfier.pth")
                
            else:
                torch.save(model.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.alpha) + "_" + str(args.energy_beta) + ".pth")
                torch.save(clsfier.state_dict(), args.save_dir + args.arch + "_" + str(args.score) + "_" + str(args.alpha) + "_" + str(args.energy_beta) + "clsfier.pth")
                

        print("Epoch [%d/%d] Loss: %.4f mAP: %.4f" % (epoch, args.n_epoch, loss.data, mAP))
    print("Best mAP:", best_mAP)
    model = best_model

    # OOD data      
    if args.ood_data == "imagenet":
        if args.dataset == "nus-wide":
            ood_root = "/data/sunyuchen/ood_datasets/dtd/images/"
            out_test_data = torchvision.datasets.ImageFolder(ood_root, transform = img_transform)
        else:
            ood_root = "/data/sunyuchen/ood_datasets/nus_ood/"
            out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "texture":
        ood_root = "/data/sunyuchen/ood_datasets/dtd/images/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform = img_transform)
    elif args.ood_data == "MNIST":
        ood_root = "/data/sunyuchen/ood_datasets/nus_ood/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "lsun":
        ood_root = "/data/sunyuchen/ood_datasets/LSUN/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == 'svhn':
        ood_root = "/data/sunyuchen/ood_datasets/svhn/"
        # out_test_data = SVHN('/data/sunyuchen/ood_datasets/nus_ood/svhn/', split='train_and_extra',
        #                   transform=torchvision.transforms.ToTensor(), download=False)
        out_test_data = SVHN(ood_root, split='test', transform = torchvision.transforms.ToTensor())
    elif args.ood_data == "places50":
        ood_root = "/data/sunyuchen/ood_datasets/Places/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "inat":
        ood_root = "/data/sunyuchen/ood_datasets/iNaturalist/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "sun50":
        ood_root = "/data/sunyuchen/ood_datasets/SUN/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    

    out_test_loader = data.DataLoader(out_test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    in_scores, in_labels = lib.get_logits(val_loader, model, clsfier, args, base_probs_tensor, name="in_test", device=device)
    out_scores, out_labels = lib.get_logits(out_test_loader, model, clsfier, args, base_probs_tensor, name="out_test", device=device)
    
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
    parser.add_argument('--ood_data', type=str, default='texture')
    parser.add_argument('--oe_data', type=str, default='imagenet')

    parser.add_argument('--n_epoch', type=int, default=250, help='# of the epochs')
    parser.add_argument('--start_epoch', type=int, default=50, help='# of the epochs')
    parser.add_argument('--n_classes', type=int, default=20, help='# of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--oe_batch_size', type=int, default=64, help='Batch Size')
    # batch_size 320 for resenet101
    parser.add_argument('--l_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--loss', type=str, default='bce', help='bce|softmax|LDAM')
    parser.add_argument('--opt', type=str, default='adam', help='adam|sgd')

    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")
    parser.add_argument('--m_in', type=float, default=25., help='default: -25. margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=7., help='default: -7. margin for out-distribution; below this value will be penalized')
    parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
    parser.add_argument('--score', default='joint', type=str, help='fine tuning mode')
    parser.add_argument('--ood', type=str, default='energy', help='which measure to use odin|M|logit|energy|msp|prob|lof|isol')
    parser.add_argument('--method', type=str, default='sum', help='which method to use max|sum')
    parser.add_argument('--k', type=int, default=50, help='bottom-k for ID')
    parser.add_argument('--alpha', default=1, type=float, help='alpha for conf loss')
    parser.add_argument('--m', default=1, type=float, help='gap between id and ood')
    parser.add_argument('--lambda_', default=0.5, type=float, help='')

    parser.add_argument('--lambda_1', default=0.07, type=float, help='for OECC')
    parser.add_argument('--lambda_2', default=0.05, type=float, help='for OECC')

    #save and load
    parser.add_argument('--load', type=bool, default=False, help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models", help='Path to load models')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    args = parser.parse_args()

    train(args)