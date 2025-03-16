import torch
import argparse
import torchvision
import lib
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from model.classifiersimple import *
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
from utils.svhn import SVHN
from utils import anom_utils

def evaluation():
    print("In-dis data: " + args.dataset)
    print("Out-dis data: " + args.ood_data)
    torch.manual_seed(0)
    np.random.seed(0)
    ###################### Setup Dataloader ######################
    # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # img_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((256, 256)),
    #     torchvision.transforms.ToTensor(),
    #     normalize,
    # ])
    # label_transform = torchvision.transforms.Compose([
    #     anom_utils.ToLabel(),
    # ])
    
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.5, 2.0)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    label_transform = torchvision.transforms.Compose([
            anom_utils.ToLabel(),
        ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    # in_dis
    if args.dataset == 'pascal':
        train_data = pascalVOCLoader('./datasets/pascal/',
                                     img_transform=img_transform, label_transform=label_transform)
        test_data = pascalVOCLoader('./datasets/pascal/', split="voc12-test",
                                    img_transform=img_transform, label_transform=None)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=img_transform, label_transform=label_transform)

    elif args.dataset == 'coco':
        train_data = cocoloader("/data/sunyuchen/datasets/COCO2014_pro/",
                             img_transform = img_transform, label_transform = label_transform)
        val_data = cocoloader('/data/sunyuchen/datasets/COCO2014_pro/', split="multi-label-val2014",
                            img_transform=val_transform, label_transform=label_transform)
        test_data = cocoloader('/data/sunyuchen/datasets/COCO2014_pro/', split="test",
                               img_transform=val_transform, label_transform=None)

    elif args.dataset == "nus-wide":
        train_data = nuswideloader("./datasets/nus-wide/",
                            img_transform = img_transform, label_transform = label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                            img_transform = val_transform, label_transform = label_transform)
        test_data = nuswideloader("./datasets/nus-wide/", split="test",
                            img_transform = val_transform, label_transform = label_transform)

    else:
        raise AssertionError

    args.n_classes = train_data.n_classes
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    in_test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=False)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=False)

    label_npy_file = args.load_model + args.dataset + "_label_sum.npy"
    print(label_npy_file, os.path.exists(label_npy_file))
    cls_num_list = torch.zeros((1, args.n_classes), dtype=torch.int)
    if not os.path.exists(label_npy_file):
        with torch.no_grad():
            for i, (_, labels) in tqdm(enumerate(train_loader)):
                labels = labels.float()
                cls_num_list = cls_num_list + torch.sum(labels, dim=0)

        cls_num_list = cls_num_list.numpy()
        os.makedirs(args.load_model, exist_ok = True)
        np.save(args.load_model + args.dataset + "_label_sum", cls_num_list)
        cls_num_list = cls_num_list[0].tolist()

    else:
        cls_num_list = np.load(label_npy_file)[0].tolist()

    base_probs = []
    for i in range(len(cls_num_list)):
        base_probs.append(cls_num_list[i] / np.array(cls_num_list).sum())
    base_probs_tensor = torch.tensor(base_probs).cuda()
    print(base_probs_tensor.sum())
    # OOD data
    if args.ood_data == "imagenet":
        # if args.dataset == "nus-wide":
        ood_root = "/data/sunyuchen/ood_datasets/nus_ood/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "imagenet22":
        ood_root = "/data/sunyuchen/ood_datasets/ImageNet-22K/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "texture":
        ood_root = "/data/sunyuchen/ood_datasets/dtd/images/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "MNIST":
        gray_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize
        ])
        out_test_data = torchvision.datasets.MNIST('/data/sunyuchen/ood_datasets/MNIST/',
                       train=False, transform=gray_transform, download=True)
    elif args.ood_data == "lsun":
        ood_root = "/data/sunyuchen/ood_datasets/LSUN/"
        out_test_data = torchvision.datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == 'svhn':
        ood_root = "/data/sunyuchen/ood_datasets/svhn/"
        # out_test_data = SVHN('/data/sunyuchen/ood_datasets/nus_ood/svhn/', split='train_and_extra',
        #                   transform=torchvision.transforms.ToTensor(), download=False)
        out_test_data = SVHN(ood_root, split='test', transform=img_transform)
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

    ###################### Load Models ######################
    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
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

    print(args.load_model)
    print(args.load_model + args.arch + "_" + str(args.score) + ".pth")
    print(args.load_model + args.arch + "_" + str(args.score) + 'clsfier' + ".pth")

    if args.score == "normal":
        model.load_state_dict(torch.load(args.load_model + args.arch + "_" + str(args.score) + ".pth"))
        clsfier.load_state_dict(torch.load(args.load_model + args.arch + "_" + str(args.score) + 'clsfier' + ".pth"))
    elif args.score == "OEML":
        model.load_state_dict(torch.load(args.load_model + "resnet50_OEML.pth"))
        clsfier.load_state_dict(torch.load(args.load_model + "resnet50_OEMLclsfier.pth"))
    elif args.score == "energyML":
        print("energyML")
        model.load_state_dict(torch.load(args.load_model + "resnet50_energyML_1114.1_0.01_50.pth"))
        clsfier.load_state_dict(torch.load(args.load_model + "resnet50_energyML_1114.1_0.01_50clsfier.pth"))
    else:
        model.load_state_dict(torch.load(args.load_model + "resnet50_joint_1.0_0.1.pth"))
        clsfier.load_state_dict(torch.load(args.load_model + "resnet50_joint_1.0_0.1clsfier.pth"))
    # model.load_state_dict(torch.load(args.load_model + "resnet50_joint_1.0_0.1_sun50.pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + "resnet50_joint_1.0_0.1_sun50clsfier.pth"))
    # model.load_state_dict(torch.load(args.load_model + "resnet50_joint_bceoe155.pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + "resnet50_joint_bceoe155clsfier.pth"))
    # model.load_state_dict(torch.load(args.load_model + "resnet50_normal_normal.pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + "resnet50_normal_normalclsfier.pth"))

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        clsfier = nn.DataParallel(clsfier, device_ids=device_ids)
    # if torch.cuda.device_count() > 1:
    #     print("Using",torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    #     clsfier = nn.DataParallel(clsfier)

    print("model loaded!")

    # freeze the batchnorm and dropout layers
    model.eval()
    clsfier.eval()
    ###################### Compute Scores ######################
    if args.ood == "odin":
        print("Using temperature", args.T, "noise", args.noise)
        in_scores, in_labels = lib.get_odin_scores(val_loader, model, clsfier, args.method,
                                        args.T, args.noise, args, "in_test", device)
        out_scores, out_labels = lib.get_odin_scores(out_test_loader, model, clsfier, args.method,
                                         args.T, args.noise, args, "out_test", device)
        # lib.get_id_auc(in_scores, out_scores, in_labels, args)
    elif args.ood == "M":
        ## Feature Extraction
        temp_x = torch.rand(2, 3, 256, 256)
        temp_x = Variable(temp_x.to(device))
        temp_list = lib.model_feature_list(model, clsfier, temp_x, args.arch)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        print('get sample mean and covariance')
        sample_mean, precision = lib.sample_estimator(model, clsfier, args.n_classes,
                                                      feature_list, train_loader, device)
        # Only use the
        pack = (sample_mean, precision)
        print("Using noise", args.noise)
        in_scores, in_labels = lib.get_Mahalanobis_score(model, clsfier, val_loader, pack,
                                              args.noise, args.n_classes, args.method, args, "in_test", device)
        out_scores, out_labels = lib.get_Mahalanobis_score(model, clsfier, out_test_loader, pack,
                                               args.noise, args.n_classes, args.method, args, "out_test", device)
        # lib.get_id_auc(in_scores, out_scores, in_labels, args)
    # elif args.ood == "gradnorm":
    #     in_scores, in_labels = lib.iterate_data_gradnorm(val_loader, model, clsfier, args.temperature_gradnorm, args.n_classes, args, "in_test", device)
    #     out_scores, out_labels = lib.iterate_data_gradnorm(out_test_loader, model, clsfier, args.temperature_gradnorm, args.n_classes, args, "out_test", device)
    # elif args.ood == "logsumexp":
    #     in_scores, in_right_score, in_wrong_score, in_labels = lib.get_energy_scores(model, clsfier, val_loader, args, "in_test")
    #     out_scores, out_right_score, out_wrong_score, out_labels = lib.get_energy_scores(model, clsfier, in_test_loader, args, "out_test")
    #     lib.get_id_auc(in_scores, out_scores, in_labels, args)
    elif args.ood == "vim":
        in_scores, out_scores = lib.get_logits_vim(train_loader, val_loader, out_test_loader, model, clsfier, args, device)

    else:
        in_scores, in_labels = lib.get_logits(val_loader, model, clsfier, args, name="in_test", device=device)
        out_scores, out_labels = lib.get_logits(out_test_loader, model, clsfier, args, name="out_test", device=device)
        # lib.get_score(val_loader, model, clsfier, args, name="in_test", device=device)
        # lib.get_score(out_test_loader, model, clsfier, args, name="out_test", device=device)
        # print(np.max(out_scores))
        # print("")
        # print(S_ood[0])
        # print(S_id[0])
        # print(torch.norm((S_ood[:10] - S_id[:10])))
        
        # lib.get_id_distribution(in_scores, in_labels, args, t=2)

        if args.ood == "lof":
            in_scores, _ = lib.get_logits(in_test_loader, model, clsfier, args, name="in_test")
            val_scores, _ = lib.get_logits(val_loader, model, clsfier, args, name="in_val")
            scores = lib.get_localoutlierfactor_scores(val_scores, in_scores, out_scores)
            in_scores = scores[:len(in_scores)]
            out_scores = scores[-len(out_scores):]

            # lib.get_id_auc(val_scores, out_scores, in_labels, args)

        if args.ood == "isol":
            val_scores, _ = lib.get_logits(val_loader, model, clsfier, args, name="in_val")
            scores = lib.get_isolationforest_scores(in_test_loader, in_scores, out_scores)
            in_scores = scores[:len(in_scores)]
            out_scores = scores[-len(out_scores):]

        # lib.get_id_auc(in_scores, out_scores, in_labels, args)
        # lib.get_id_auc(val_scores, out_scores, in_labels, args)
    ###################### Measure ######################
    anom_utils.get_and_print_results(in_scores, out_scores, args.ood, args.method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # ood measures
    parser.add_argument('--ood', type=str, default='energy',
                        help='which measure to use odin|M|logit|energy|msp|prob|lof|isol')
    parser.add_argument('--method', type=str, default='sum',
                        help='which method to use max|sum')
    # dataset
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--ood_data', type=str, default='imagenet')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch Size')
    parser.add_argument('--n_classes', type=int, default=20, help='# of classes')
    # save and load
    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")
    parser.add_argument('--load_model', type=str, default="./saved_models/",
                        help='Path to load models')
    # input pre-processing
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--score', default='joint', type=str, help='fine tuning mode')
    
    # parameters for other methods
    parser.add_argument('--threshold', default=1, type=float, help='threshold for ReAct')
    parser.add_argument('--temperature_gradnorm', default=1, type=int, help='temperature scaling for GradNorm')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')

    args = parser.parse_args()
    args.load_model += args.dataset + '/'

    args.save_path += args.dataset + '/' + args.ood_data + '/' + args.arch + '/'
    evaluation()