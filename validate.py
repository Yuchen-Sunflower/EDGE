import torch
import argparse
import torchvision
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

from utils.dataloader.coco_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.pascal_voc_loader import *
from utils.anom_utils import ToLabel
import pandas as pd
import os

from model.classifiersimple import *

print("Using", torch.cuda.device_count(), "GPUs")

def validate(args, model, clsfier, val_loader, device):
    model.eval()
    clsfier.eval()

    gts = {i:[] for i in range(0, args.n_classes)}
    preds = {i:[] for i in range(0, args.n_classes)}
    # feats = torch.zeros((args.batch_size, 2048)).to(device)
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(val_loader)):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device).float())
            feat = model(images)
            feat = F.relu(feat, inplace=True)
            outputs = clsfier(feat)
            outputs = torch.sigmoid(outputs)
            # if images.shape[0] == args.batch_size:
                # feats = feats + feat.view(feat.shape[0], -1)

            pred = outputs.squeeze().data.cpu().numpy()
            gt = labels.squeeze().data.cpu().numpy()
            
            for label in range(0, args.n_classes):
                gts[label].extend(gt[:,label])
                preds[label].extend(pred[:,label])

    # feats = feats / (i + 1)
    # U_test, S_test, V_test = torch.svd(feats)
    # print(torch.mean(torch.abs(S_test - S_id)))

    FinalMAPs = []
    for i in range(0, args.n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i])
        FinalMAPs.append(metrics.auc(recall, precision))
    # print(FinalMAPs)

    return np.mean(FinalMAPs)

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


# def validate(args, model, clsfier, val_loader, device):
#     model.eval()
#     clsfier.eval()

#     gts = {i:[] for i in range(0, args.n_classes)}
#     preds = {i:[] for i in range(0, args.n_classes)}
#     # gts = []
#     # preds = []
#     in_labels = np.empty([0, args.n_classes])
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = Variable(images.to(device))
#             labels = Variable(labels.to(device).float())
#             feat = model(images)
#             feat = F.relu(feat, inplace=True)
#             outputs = clsfier(feat)
#             outputs = torch.sigmoid(outputs)

#             pred = outputs.squeeze().data.cpu().numpy()
#             gt = labels.squeeze().data.cpu().numpy()

#             in_labels = np.vstack((in_labels, gt))
            
#             # gts.append(labels.cpu().detach())
#             # preds.append(outputs.cpu().detach())
#             for label in range(0, args.n_classes):
#                 gts[label].extend(gt[:,label])
#                 preds[label].extend(pred[:,label])

#     # mAP_score = mAP(targs=torch.cat(gts).numpy(), preds=torch.cat(preds).numpy())
#     # return mAP_score
#     class_counts = in_labels.sum(0)
#     sorted_indices = np.argsort(-class_counts)
#     # print(class_counts)
#     FinalMAPs = []
#     # FinalAUC = []
#     for i in range(0, args.n_classes):
#         precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i])
#         # auroc = metrics.roc_auc_score(gts[i], preds[i])
#         # FinalAUC.append(auroc)
#         FinalMAPs.append(metrics.auc(recall, precision))
#     # print(FinalMAPs)
#     # return np.mean(FinalAUC)
#     AP_sort = np.array(FinalMAPs)[sorted_indices]
#     # print(AP_sort)
#     split_size = len(AP_sort) // 8
#     means = [np.mean(AP_sort[i * split_size: (i + 1) * split_size]) for i in range(7)]
#     means.append(np.mean(AP_sort[7 * split_size:]))

#     df = pd.DataFrame({
#         "ap": means
#     })
#     save_result = f'./score_result/{args.dataset}/{args.arch}'
#     if not os.path.exists(save_result):
#         os.makedirs(save_result)
#     csv_ap = f'./score_result/{args.dataset}/{args.arch}/ap_sort.csv'
#     df.to_csv(csv_ap, index=False)
#     return np.mean(FinalMAPs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet101', help='Architecture to use')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', help='Dataset to use [\'pascal, coco, nus-wide\']')
    parser.add_argument('--load_model', type=str, default="./saved_models/", help='Path to load models')
    parser.add_argument('--batch_size', nargs='?', type=int, default=20, help='Batch Size')
    parser.add_argument('--n_classes', nargs='?', type=int, default=20)
    parser.add_argument('--score', default='joint', type=str, help='joint|normal|react')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    args = parser.parse_args()

    # Setup Dataloader
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    label_transform = torchvision.transforms.Compose([
        ToLabel(),
    ])

    if args.dataset == 'pascal':
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val", img_transform=img_transform, label_transform=label_transform)
    elif args.dataset == 'coco':
        val_data = cocoloader('/data/sunyuchen/datasets/COCO2014_pro/', split="multi-label-val2014", img_transform=img_transform, label_transform=label_transform)
    elif args.dataset == "nus-wide":
        val_data = nuswideloader("./datasets/nus-wide/", split="val", img_transform = img_transform, label_transform = label_transform)
    else:
        raise AssertionError

    args.n_classes = val_data.n_classes
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=False)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
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
    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, args.n_classes)

    print(args.load_model)
    # print(args.load_model + args.dataset + "/" + args.arch + "_" + str(args.score) + ".pth")
    # print(args.load_model + args.dataset + "/" + args.arch + "_" + str(args.score) + 'clsfier' + ".pth")
    # model.load_state_dict(torch.load(args.load_model + args.dataset + "/" + args.arch + "_" + str(args.score) + ".pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + args.dataset + "/" + args.arch + "_" + str(args.score) + 'clsfier' + ".pth"))
    # model.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_joint_1.0_0.1_places50.pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_joint_1.0_0.1_places50clsfier.pth"))
    # model.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_energyML_1114.0_0.01_50.pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_energyML_1114.0_0.01_50clsfier.pth"))
    # model.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_OEML.pth"))
    # clsfier.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_OEMLclsfier.pth"))
    model.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_normal.pth"))
    clsfier.load_state_dict(torch.load(args.load_model + args.dataset + "/" + "resnet50_normalclsfier.pth"))
    print("model loaded!")

    device_ids = list(map(int, args.device_id.split(',')))
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print("Available device = ", str(device_ids))
    model.to(device)
    clsfier.to(device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        clsfier = nn.DataParallel(clsfier, device_ids=device_ids)

    mAP = validate(args, model, clsfier, val_loader, device)
    print("mAP on validation set: %.4f" % (mAP * 100))