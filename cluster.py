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

# def extract_features_eps(dataset, model):
#     model.eval()
#     features_list = []
#     with torch.no_grad():
#         for data, _ in tqdm(dataset):
#             data = (data + 1e-5).cuda()
#             features = model(data).view(data.shape[0], -1)
#             num_samples = features.shape[0]
#             select_count = int(0.005 * num_samples)
#             selected_indices = np.random.choice(num_samples, select_count, replace=False)
#             selected_features = features[selected_indices]
            
#             features_list.append(selected_features.cpu())
#     return torch.vstack(features_list)

# def compute_mean_covariance(features):
#     mean = torch.mean(features, dim=0)
#     features_centered = features - mean
#     covariance = features_centered.t().mm(features_centered) / features.size(0)
#     return mean, covariance

def main(args):

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

    if args.oe_data == "imagenet":
        oe_root = "/data/sunyuchen/ood_datasets/nus_ood/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
        # oe_root = "/data/sunyuchen/ood_datasets/ImageNet-22K/"
        # out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "MNIST":
        oe_root = "/data/sunyuchen/ood_datasets/MNIST/"
        out_test_data = torchvision.datasets.MNIST(oe_root, train=False, transform=gray_transform, download=True)
    elif args.oe_data == "texture":
        oe_root = "/data/sunyuchen/ood_datasets/dtd/images/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform = img_transform)
    elif args.oe_data == "lsun":
        oe_root = "/data/sunyuchen/ood_datasets/LSUN/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == 'svhn':
        oe_root = "/data/sunyuchen/ood_datasets/svhn/"
        # out_test_data = SVHN('/data/sunyuchen/ood_datasets/nus_ood/svhn/', split='train_and_extra',
        #                   transform=torchvision.transforms.ToTensor(), download=False)
        # out_test_data = SVHN(oe_root, split='train_and_extra', transform = torchvision.transforms.ToTensor())
        out_test_data = SVHN(oe_root, split='train_and_extra', transform=img_transform)
    elif args.oe_data == "places50":
        oe_root = "/data/sunyuchen/ood_datasets/Places/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "iNaturalist":
        oe_root = "/data/sunyuchen/ood_datasets/iNaturalist/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)
    elif args.oe_data == "sun50":
        oe_root = "/data/sunyuchen/ood_datasets/SUN/"
        out_test_data = torchvision.datasets.ImageFolder(oe_root, transform=img_transform)

    trainloader = data.DataLoader(train_data, batch_size=200, num_workers=8, shuffle=True, pin_memory=True)
    out_test_loader = data.DataLoader(out_test_data, batch_size=200, num_workers=8, shuffle=True, pin_memory=True)
    out_test_loader.dataset.offset = np.random.randint(len(out_test_loader.dataset))

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[:-1])
        # clsfier = clssimp(2048, args.n_classes)
        # binary_clsfier = clssimp(2048, 1)
    elif args.arch == "resnet50":
        orig_resnet = torchvision.models.resnet50(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        # clsfier = clssimp(num_ftrs, args.n_classes)
        # binary_clsfier = clssimp(num_ftrs, 1)
    elif args.arch == "resnet34":
        orig_resnet = torchvision.models.resnet34(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        # clsfier = clssimp(num_ftrs, args.n_classes)

    elif args.arch == "resnet18":
        orig_resnet = torchvision.models.resnet18(pretrained=True)
        features = list(orig_resnet.children())
        num_ftrs = orig_resnet.fc.in_features
        model= nn.Sequential(*features[:-1])
        # clsfier = clssimp(num_ftrs, args.n_classes)

    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        # clsfier = clssimp(1024, args.n_classes)

    # model = WideResNet(40, 10, 2, dropRate=0.3).cuda()
    # model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的全连接层
    print(f"./saved_models/{args.dataset}/{args.arch}_{args.score}.pth")
    model.cuda()
    model.load_state_dict(torch.load(f"./saved_models/{args.dataset}/{args.arch}_{args.score}.pth"))

# def extract_features(dataloader, model):
#     model.eval()
#     features_list = []
#     labels_list = []
#     min_size = float('inf')  # 初始化为无穷大，用于找到最小批次的尺寸

#     with torch.no_grad():
#         for data, labels in tqdm(dataloader):
#             data = data.cuda()
#             features = model(data).squeeze(-1).squeeze(-1)
#             min_size = min(min_size, features.size(0))  # 更新最小批次尺寸
#             features_list.append(features.cpu())
#             labels_list.append(labels.cpu())

#     # 根据最小批次尺寸调整特征尺寸
#     features_list = [features[:min_size] for features in features_list]
#     labels_list = [labels[:min_size] for labels in labels_list]

#     return torch.vstack(features_list), torch.vstack(labels_list)


    nuswide_features = extract_features(trainloader, model)
    # nuswide_features_eps = extract_features_eps(trainloader, model)
    imagenet_features = extract_features(out_test_loader, model)

    # nuswide_mean, nuswide_cov = compute_mean_covariance(nuswide_features)
    # imagenet_mean, imagenet_cov = compute_mean_covariance(imagenet_features)

    # print(nuswide_mean)
    # # print(imagenet_mean.shape)

    # # print(nuswide_cov.shape)
    # # print(imagenet_cov.shape)

    # all_features = torch.vstack([nuswide_features, imagenet_features, nuswide_features_eps])
    all_features = torch.vstack([nuswide_features, imagenet_features])
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(all_features)
    df = pd.DataFrame(embedding, columns = ['x', 'y'])
    df['label'] = [args.dataset if i < len(nuswide_features) else args.oe_data for i in range(len(embedding))]
    df.to_csv(f'{args.dataset}_{args.oe_data}_{args.score}_embedding.csv', index=False)
    print(embedding)

    # 可视化
    plt.scatter(embedding[:len(nuswide_features), 0], embedding[:len(nuswide_features), 1], label=args.dataset, s=2)
    # plt.scatter(embedding[len(nuswide_features):len(nuswide_features) + len(imagenet_features), 0], embedding[len(nuswide_features):len(nuswide_features) + len(imagenet_features), 1], label=oe, s=1)
    # plt.scatter(embedding[len(nuswide_features) + len(imagenet_features):, 0], embedding[len(nuswide_features) + len(imagenet_features):, 1], label=f"{dataset}-eps", s=1)
    plt.scatter(embedding[len(nuswide_features):, 0], embedding[len(nuswide_features):, 1], label=args.oe_data, s=2)
    plt.legend(fontsize=18, loc="upper right", markerscale=8)
    plt.xticks([])  # 移除x轴刻度
    plt.yticks([])  # 移除y轴刻度
    plt.savefig(f'./umap/{args.dataset}_{args.oe_data}_{args.score}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)  # 保存为PNG格式，分辨率为300dpi

    # id_features, id_labels = extract_features(trainloader, model)
    # ood_features, ood_labels = extract_features(out_test_loader, model)
    # all_features = torch.vstack([id_features, ood_features])

    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(id_features)
    # embedding_ood = reducer.fit_transform(ood_features)
    # plt.figure(figsize=(10, 8))
    # classes = [28, 68, 25, 18, 60, 12, 11, 7, 53, 69]
    # plt.scatter(embedding[:, 0], embedding[:, 1], label=dataset, s=2)
    # for class_label in tqdm(classes):  # Assuming you have 10 classes
    # # for class_label in range(20):  # Assuming you have 10 classes
    #     indices = [i for i, labels in enumerate(id_labels) if labels[class_label] == 1]
    #     plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f'Class {class_label}', s=2)
    # plt.scatter(embedding_ood[:, 0], embedding_ood[:, 1], label=oe, s=2)

    # plt.title(f'UMAP visualization for {dataset} classes')
    # plt.legend()
    # plt.savefig(f'umap_{dataset}_{oe}_{train}.png', dpi=300)
    # plt.show()

# def train(epoch):
#     net.train()  # enter train mode
#     loss_avg = 0.0
#     total_data_samples = 0

#     for data, target in train_loader:
#         data, target = data.cuda(), target.cuda()

#         # forward
#         x, output = net.forward_virtual(data)

#         # energy regularization
#         lr_reg_loss = torch.zeros(1).cuda()[0]
        
#         if total_data_samples < args.sample_number:
#             if total_data_samples == 0:
#                 data_samples = output.detach()
#             else:
#                 data_samples = torch.cat((data_samples, output.detach()), 0)

#             total_data_samples += len(target)
#         elif total_data_samples == args.sample_number:
#             if epoch >= args.start_epoch:
#                 # center the data
#                 X = data_samples - data_samples.mean(0)
#                 # calculate covariance
#                 temp_precision = torch.mm(X.t(), X) / len(X)
#                 temp_precision += 0.0001 * eye_matrix

#                 # Use multivariate normal distribution with the computed covariance
#                 mean_embed = data_samples.mean(0)
#                 new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
#                     mean_embed, covariance_matrix=temp_precision)
#                 negative_samples = new_dis.rsample((args.sample_from,))
#                 prob_density = new_dis.log_prob(negative_samples)

#                 cur_samples, index_prob = torch.topk(-prob_density, args.select)
#                 ood_samples = negative_samples[index_prob]

#                 if len(ood_samples) != 0:
#                     energy_score_for_fg = log_sum_exp(x, 1)
#                     predictions_ood = net.fc(ood_samples)
#                     energy_score_for_bg = log_sum_exp(predictions_ood, 1)

#                     input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
#                     labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
#                                                torch.zeros(len(ood_samples)).cuda()), -1)

#                     criterion = torch.nn.CrossEntropyLoss()
#                     output1 = logistic_regression(input_for_lr.view(-1, 1))
#                     lr_reg_loss = criterion(output1, labels_for_lr.long())

#                     if epoch % 5 == 0:
#                         print(lr_reg_loss)

#         # backward
#         optimizer.zero_grad()
#         loss = F.cross_entropy(x, target)
#         loss += args.loss_weight * lr_reg_loss
#         loss.backward()

#         optimizer.step()
#         scheduler.step()

#         # exponential moving average
#         loss_avg = loss_avg * 0.8 + float(loss) * 0.2

#     state['train_loss'] = loss_avg

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
    parser.add_argument('--m_in', type=float, default=-25., help='default: -25. margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='default: -7. margin for out-distribution; below this value will be penalized')
    parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
    parser.add_argument('--score', default='joint', type=str, help='fine tuning mode')
    parser.add_argument('--ood', type=str, default='energy', help='which measure to use odin|M|logit|energy|msp|prob|lof|isol')
    parser.add_argument('--method', type=str, default='sum', help='which method to use max|sum')
    parser.add_argument('--k', type=int, default=50, help='bottom-k for ID')
    parser.add_argument('--alpha', default=1, type=float, help='alpha for conf loss')
    parser.add_argument('--m', default=1, type=float, help='gap between id and ood')

    #save and load
    parser.add_argument('--load', type=bool, default=False, help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models", help='Path to load models')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    args = parser.parse_args()
    # params = nni.get_next_parameter()
    main(args)