import os
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from utils import anom_utils
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_id_auc(in_scores, out_scores, in_labels, args):
    all_instance_labels = in_labels
    print(in_labels.shape)
    # print(all_instance_labels.sum(axis=0).shape)
    all_labels_argsort = np.argsort(all_instance_labels.sum(axis=0))[::-1]
    file_path = f'./score_result/{args.dataset}/label_argsort.npy'
    if not os.path.exists(file_path):
        print(all_labels_argsort)
        np.save(f'./score_result/{args.dataset}/label_argsort', all_labels_argsort)
    
    if args.dataset == "nus-wide":
        head_position = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60]
        tail_position = [1, 2, 3, 5, 10, 20, 40, 60, 70, 75, 78, 79, 80]
    elif args.dataset == "pascal":
        head_position = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        tail_position = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    elif args.dataset == "coco":
        head_position = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60]
        tail_position = [1, 2, 3, 5, 10, 20, 40, 60, 70, 75, 78, 79]

    df = pd.DataFrame(columns=["FPR95", "AUROC", "AUPR"])

    for j in head_position:
        all_labels_argsort_top = all_labels_argsort[:j]
    # for j in tail_position:
    #     all_labels_argsort_top = all_labels_argsort[-j:]
        sub_labels = all_instance_labels[:, all_labels_argsort_top]
        mask = np.all(sub_labels == 0, axis=1)
        # print((mask == True).sum())
        # all_labels = all_instance_labels[mask, :]
        # print(in_scores.shape)
        # print(mask.shape)
        scores_j = in_scores[mask]

        # print(scores_j.mean())

        scores = np.concatenate((scores_j, out_scores))
        labels = np.zeros(scores.shape, dtype=np.int32)
        labels[:scores_j.shape[0]] += 1
        auroc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        fpr, threshould = anom_utils.fpr_and_fdr_at_recall(labels, scores, 0.95)

        new_row = {"FPR95": fpr, "AUROC": auroc, "AUPR": aupr}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(out_scores.mean())
    print(df.to_string(index=False))
    score_result_path = f'./score_result/{args.dataset}/{args.arch}/{args.ood_data}_{args.ood}_{args.method}_{args.score}.csv'
    df.to_csv(score_result_path, index=False)
    
    # for i in range(all_instance_labels.shape[1]):
    #     relevant_index = np.where(all_instance_labels[:, i] == 1)[0]
    #     relevant_scores = in_scores[relevant_index]

    #     scores = np.concatenate((relevant_scores, out_scores))
    #     labels = np.zeros(scores.shape, dtype=np.int32)
    #     labels[:relevant_scores.shape[0]] += 1
    #     if relevant_scores.shape[0] == 0:
    #         auroc = 0.5
    #     else:
    #         auroc = roc_auc_score(labels, scores)
    #     aurocs.append(auroc)

    # aurocs = np.array(aurocs)
    # auroc_sorted_by_num = aurocs[all_labels_argsort]

        # mean_auroc = []
        # print(auroc_sorted_by_num)
        # print(auroc_sorted_by_num[:j].mean())

        # for i in range(4):
        #     if i == 3 and args.dataset == "nus-wide":
        #         mean_auroc.append(auroc_sorted_by_num[i * 20: (i + 1) * 20 + 1].mean())
        #     elif i == 0:
        #         mean_auroc.append(auroc_sorted_by_num[j: (i + 1) * 20].mean())
        #     else:
        #         mean_auroc.append(auroc_sorted_by_num[i * 20: (i + 1) * 20].mean())
        # df[j] = np.array(mean_auroc)
        # print(mean_auroc)

        # mean_auroc = []
        # for i in range(8):
        #     if i == 7 and args.dataset == "nus-wide":
        #         mean_auroc.append(auroc_sorted_by_num[i * 10: (i + 1) * 10 + 1].mean())
        #     elif i == 0:
        #         mean_auroc.append(auroc_sorted_by_num[j: (i + 1) * 10].mean())
        #     else:
        #         mean_auroc.append(auroc_sorted_by_num[i * 10: (i + 1) * 10].mean())
        # df[j] = np.array(mean_auroc)
        # print(mean_auroc)

    # for i in range(8):
    #     mean_auroc.append(auroc_sorted_by_num[i * 10: (i + 1) * 10].mean())

    # for i in range(1):
    #     if i == 3 and args.dataset == "nus-wide":
    #         mean_auroc.append(auroc_sorted_by_num[i * 20: (i + 1) * 20 - 9].mean())
    #     else:
    #         mean_auroc.append(auroc_sorted_by_num[i * 20: (i + 1) * 20].mean())
    # print(mean_auroc)
    # mean_auroc = []
    # for i in range(2):
    #     if i == 7 and args.dataset == "nus-wide":
    #         mean_auroc.append(auroc_sorted_by_num[i * 10: (i + 1) * 10 - 9].mean())
    #     else:
    #         mean_auroc.append(auroc_sorted_by_num[i * 10: (i + 1) * 10].mean())
    # print(mean_auroc)

    
    # save_result = f'./score_result/{args.dataset}/{args.arch}'
    # if not os.path.exists(save_result):
    #     os.makedirs(save_result)
    # csv_logit = f'./logit_10.csv'
    # df.to_csv(csv_logit, index=False)
    
def get_id_distribution(in_scores, in_labels, args, t=0.69):
    all_index = np.arange(0, in_scores.shape[0], dtype=int)
    neg_index = np.where(in_scores < t)[0]
    pos_index = np.setdiff1d(all_index, neg_index)

    all_instance_labels = in_labels
    neg_instance_labels = in_labels[neg_index, :]
    pos_instance_labels = in_labels[pos_index, :]

    pos_sum = pos_instance_labels.sum(axis=0)
    all_sum = all_instance_labels.sum(axis=0)

    acc = pos_sum / all_sum
    # index = np.argsort(acc)
    all_labels_sort = np.sort(all_instance_labels.sum(axis=0))[::-1]
    all_labels_argsort = np.argsort(all_instance_labels.sum(axis=0))[::-1]
    pos_labels_argsort = np.argsort(all_instance_labels.sum(axis=0))[::-1]

    # all_labels_argsort_top = all_labels_argsort[:3]
    
    # sub_labels = all_instance_labels[:, all_labels_argsort_top]
    # mask = np.all(sub_labels == 0, axis=1)
    # all_instance_labels = all_instance_labels[mask, :]
    # in_scores = in_scores[mask]

    acc_sorted_by_num = acc[all_labels_argsort]
    mean_acc = []
    for i in range(8):
        mean_acc.append(acc_sorted_by_num[i * 10: (i + 1) * 10].mean())
    print(mean_acc)

def get_odin_scores(loader, model, clsfier, method, T, noise, args, name=None, device=None):
    ## get logits
    bceloss = nn.BCEWithLogitsLoss(reduction="none")
    in_labels = np.empty([0, args.n_classes])
    for i, (images, targets) in enumerate(loader):
        if device == None:
            images = Variable(images.cuda(), requires_grad=True)
        else:
            images = Variable(images.to(device), requires_grad=True)
        nnOutputs = clsfier(model(images))

        # using temperature scaling
        preds = torch.sigmoid(nnOutputs / T)
        if device == None:
            labels = torch.ones(preds.shape).cuda() * (preds >= 0.5)
        else:
            labels = torch.ones(preds.shape).to(device) * (preds >= 0.5)
        labels = Variable(labels.float())
        # print(targets)
        if name == "in_test":
            in_labels = np.vstack((in_labels, targets))
        # input pre-processing
        loss = bceloss(nnOutputs, labels)

        if method == 'max':
            idx = torch.max(preds, dim=1)[1].unsqueeze(-1)
            loss = torch.mean(torch.gather(loss, 1, idx))
        elif method == 'sum':
            loss = torch.mean(torch.sum(loss, dim=1))

        loss.backward()
        # calculating the perturbation
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).to(device) ,
                             gradient.index_select(1, torch.LongTensor([0]).to(device) ) / (0.229))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device) ,
                             gradient.index_select(1, torch.LongTensor([1]).to(device) ) / (0.224))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device) ,
                             gradient.index_select(1, torch.LongTensor([2]).to(device) ) / (0.225))
        tempInputs = torch.add(images.data, gradient, alpha=-noise)

        with torch.no_grad():
            nnOutputs = clsfier(model(Variable(tempInputs)))

            ## compute odin score
            outputs = torch.sigmoid(nnOutputs / T)

            if method == "max":
                score = np.max(to_np(outputs), axis=1)
            elif method == "sum":
                score = np.sum(to_np(outputs), axis=1)

            if i == 0:
                scores = score
            else:
                scores = np.concatenate((scores, score),axis=0)

    return scores, in_labels

def sample_estimator(model, clsfier, num_classes, feature_list, train_loader, device=None):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []

    # list_features = []
    # for i in range(num_output):
    #     temp_list = []
    #     for j in range(num_classes):
    #         temp_list.append(0)
    #     list_features.append(temp_list)

    for j in range(num_classes):
        list_features.append(0)

    idx = 0
    with torch.no_grad():
        for data, target in train_loader:
            idx += 1
            print(idx)
            if device == None:
                data = Variable(data.cuda())
                target = target.cuda()
            else:
                data = Variable(data.to(device))
                target = target.to(device)

            # output, out_features = model_feature_list(model, clsfier, data)  # output = size[batch_size, num_class]
            # get hidden features
            # for i in range(num_output):
            #     out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            #     out_features[i] = torch.mean(out_features[i].data, 2)

            out_features = model(data)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features.data, 2)

            # construct the sample matrix
            # use the training set labels(multiple) or set with the one with max prob

            for i in range(data.size(0)):
                # px = 0
                for j in range(num_classes):
                    if target[i][j] == 0:
                        continue
                    label = j
                    if num_sample_per_class[label] == 0:
                        # out_count = 0
                        # for out in out_features:
                        #     list_features[out_count][label] = out[i].view(1, -1)
                        #     out_count += 1

                        list_features[label] = out_features[i].view(1, -1)
                    else:
                        # out_count = 0
                        # for out in out_features:
                        #     list_features[out_count][label] \
                        #         = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        #     out_count += 1

                        list_features[label] = torch.cat((list_features[label],
                                                          out_features[i].view(1, -1)), 0)
                    num_sample_per_class[label] += 1

    # sample_class_mean = []
    # out_count = 0
    # for num_feature in feature_list:
    #     temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
    #     for j in range(num_classes):
    #         temp_list[j] = torch.mean(list_features[out_count][j], 0)
    #     sample_class_mean.append(temp_list)
    #     out_count += 1

    num_feature = feature_list[-1]
    if device == None:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
    else:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
    for j in range(num_classes):
        temp_list[j] = torch.mean(list_features[j], 0)
    sample_class_mean = temp_list

    # precision = []
    # for k in range(num_output):
    #     X = 0
    #     for i in range(num_classes):
    #         if i == 0:
    #             X = list_features[k][i] - sample_class_mean[k][i]
    #         else:
    #             X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
    #
    #     # find inverse
    #     group_lasso.fit(X.cpu().numpy())
    #     temp_precision = group_lasso.precision_
    #     temp_precision = torch.from_numpy(temp_precision).float().cuda()
    #     precision.append(temp_precision)

    X = 0
    for i in range(num_classes):
        if i == 0:
            X = list_features[i] - sample_class_mean[i]
        else:
            X = torch.cat((X, list_features[i] - sample_class_mean[i]), 0)
    # find inverse
    group_lasso.fit(X.cpu().numpy())
    temp_precision = group_lasso.precision_
    if device == None:
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
    else:
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
    precision = temp_precision

    return sample_class_mean, precision


def get_Mahalanobis_score(model, clsfier, loader, pack, noise, num_classes, method, args, name=None, device=None):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    sample_mean, precision = pack
    model.eval()
    clsfier.eval()
    Mahalanobis = []
    in_labels = np.empty([0, args.n_classes])
    for i, (data, target) in enumerate(loader):
        if device == None:
            data = Variable(data.cuda(), requires_grad=True)
        else:
            data = Variable(data.to(device), requires_grad=True)

        if name == "in_test":
            in_labels = np.vstack((in_labels, target))

        # out_features = model_penultimate_layer(model, clsfier, data)
        out_features = model(data)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)  # size(batch_size, F)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean.index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision)), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if device == None:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.229))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.224))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.225))
            tempInputs = torch.add(data.data, gradient, alpha=-noise)
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).to(device),
                                 gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.229))
            gradient.index_copy_(1, torch.LongTensor([1]).to(device),
                                 gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.224))
            gradient.index_copy_(1, torch.LongTensor([2]).to(device),
                                 gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.225))
            tempInputs = torch.add(data.data, gradient, alpha=-noise)
        #noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        with torch.no_grad():
            # noise_out_features = model_penultimate_layer(model, clsfier, Variable(tempInputs))
            noise_out_features = model(Variable(tempInputs))
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
        # noise_gaussion_score size([batch_size, n_classes])

        if method == "max":
            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        elif method == "sum":
            noise_gaussian_score = torch.sum(noise_gaussian_score, dim=1)

        Mahalanobis.extend(to_np(noise_gaussian_score))

    return np.array(Mahalanobis), in_labels

def get_energy_scores(model, clsfier, loader, args, name, device=None):
    _score = []
    _right_score = []
    _wrong_score = []

    model.eval()
    clsfier.eval()
    in_labels = np.empty([0, args.n_classes])
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if device == None:
                data = data.cuda()
            else:    
                data = data.to(device)
            output = model(data)
            output = clsfier(output)
            smax = to_np(F.softmax(output, dim=1))
            if name == "in_test":
                in_labels = np.vstack((in_labels, target))
            # if args.use_xent:
            #     _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            # else:
            #     if args.score == 'energy':
            #         _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
            #     else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
            #         _score.append(-np.max(smax, axis=1))
            _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            # if args.use_xent:
                # _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                # _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
            # else:
            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy(), in_labels


def model_feature_list(model, clsfier, x, arch):
    out_list = []
    # features = list(model.children())
    if arch == "resnet101":
        out = model[:4](x)
        out_list.append(out)
        out = model[4](out)
        out_list.append(out)
        out = model[5](out)
        out_list.append(out)
        out = model[6](out)
        out_list.append(out)
        out = model[7](out)
        out_list.append(out.data)
    elif arch == "resnet50":
        out = model[:4](x)
        out_list.append(out)
        out = model[4](out)
        out_list.append(out)
        out = model[5](out)
        out_list.append(out)
        out = model[6](out)
        out_list.append(out)
        out = model[7](out)
        out_list.append(out.data)
    elif arch == "densenet":
        out = model[:4](x)
        out_list.append(out)
        out = model[4:6](out)
        out_list.append(out)
        out = model[6:8](out)
        out_list.append(out)
        out = model[8:10](out)
        out_list.append(out)
        out = model[10:](out)
        out_list.append(out.data)
    return clsfier(out), out_list

# def get_energy_distribution(in_score, out_score, args, device):


def get_logits(loader, model, clsfier, args, name=None, device=None):
    npy_file = args.save_path + str(args.score) + "_" + name + ".npy"
    label_npy_file = args.save_path + str(args.score) + "_" + "label.npy"
    
    print(npy_file, os.path.exists(npy_file))
    print(label_npy_file, os.path.exists(label_npy_file))
    in_labels = np.empty([0, args.n_classes])

    if not (os.path.exists(npy_file) and os.path.exists(label_npy_file)):
        logits_np = np.empty([0, args.n_classes])
        if device == None:
            feats = torch.zeros((args.batch_size, 2048)).cuda()
        else:
            feats = torch.zeros((args.batch_size, 2048)).to(device)
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                if device == None:
                    images = Variable(images.cuda())
                else:
                    images = Variable(images.to(device))
                nnOutputs = model(images)
                feat = nnOutputs.view(nnOutputs.shape[0], -1)
                if images.shape[0] == args.batch_size:
                    # U, S, V = torch.svd(feat)
                    # print(S)
                    feats = feats + nnOutputs.view(nnOutputs.shape[0], -1)

                if args.ood == 'react':
                    mask = nnOutputs.view(nnOutputs.shape[0], nnOutputs.shape[1], -1).mean(2) < args.threshold
                    nnOutputs = mask[:, :, None, None] * nnOutputs
                nnOutputs = clsfier(nnOutputs)
                # nnOutputs = torch.sigmoid(nnOutputs)

                nnOutputs_np = to_np(nnOutputs.squeeze())
                logits_np = np.vstack((logits_np, nnOutputs_np))
                if name == "in_test":
                    in_labels = np.vstack((in_labels, labels.cpu().numpy()))

        feats = feats / i
        # U_test, S_test, V_test = torch.svd(feats)
        # print(S_test)
        # print(torch.mean(torch.abs(S_test - S_id)))
        os.makedirs(args.save_path, exist_ok = True)
        # np.save(args.save_path + str(args.score) + "_" + name, logits_np)
        # if name == "in_test":
        #     np.save(args.save_path + str(args.score) + "_label", in_labels)

    else:
        logits_np = np.load(npy_file)
        in_labels = np.load(label_npy_file)
        # print(in_labels[0])
    ## Compute the Score
    if device == None:
        logits = torch.from_numpy(logits_np).cuda()
        in_labels = torch.from_numpy(in_labels).cuda()
    else:
        logits = torch.from_numpy(logits_np).to(device)
        in_labels = torch.from_numpy(in_labels).to(device)
    outputs = torch.sigmoid(logits)
    print(logits.mean(1))
    print(torch.max(logits, dim=1)[0].mean())
    print(torch.min(logits, dim=1)[0].mean())
    if args.ood == "logit":
        if args.method == "max": scores = np.max(logits_np, axis=1)
        if args.method == "sum": scores = np.sum(logits_np, axis=1)
    elif args.ood == "energy":
        # E_f = torch.log(1 + torch.exp(outputs))
        # if args.score == "energy" or args.score == "OE":
        #     E_f = -torch.log(1 + torch.exp(logits))
        # else:
        E_f = torch.log(1 + torch.exp(logits))
        # logits = F.relu(logits)
        # E_f = torch.exp(logits - 1)
        #  + 0.1 * torch.log(base_probs_tensor.view(1, -1))
        # E_f = 1 / (1 + torch.exp(-logits))
        if args.method == "max": scores = to_np(torch.max(E_f, dim=1)[0])
        if args.method == "sum": 
            scores = to_np(torch.sum(E_f, dim=1))
            indices = np.argsort(scores)
            if name == "in_test":
                print(outputs[indices[:10]])
            else:
                print(outputs[indices[-10:]])
        if args.method == "topk": scores = to_np(torch.sum(torch.topk(E_f, k=1, dim=1)[0], dim=1))
    elif args.ood == "prob":
        if args.method == "max": scores = np.max(to_np(outputs), axis=1)
        if args.method == "sum": scores = np.sum(to_np(outputs), axis=1)
    elif args.ood == "msp":
        outputs = F.softmax(logits, dim=1)
        scores = np.max(to_np(outputs), axis=1)
    elif args.ood == "react":
        scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    # elif args.ood == "react+joint":
    #     E_f = torch.log(1 + torch.exp(logits))
    #     scores = to_np(torch.sum(E_f, dim=1))
    elif args.ood == "energyMC":
        scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    # elif args.ood == "piece":
    #     _, index = torch.topk(-base_probs_tensor, k=len(base_probs_tensor)//2)
    #     mask = torch.zeros_like(base_probs_tensor, dtype=int)
    #     mask[index] = 1

        # print(mask)
        # print(mask.shape)
        # print(mask.sum())
        # print((base_probs_tensor * mask).sum())
    else:
        scores = logits_np

    return scores, to_np(in_labels)


def get_logits_vim(train_loader, val_loader, ood_loader, model, clsfier, args, device=None):
   
    in_labels = np.empty([0, args.n_classes])
    feature_id_train = []
    feature_id_val = []
    feature_ood = []
    logits_np = np.empty([0, args.n_classes])

    w = clsfier.cls.weight.data.detach().cpu().numpy()
    b = clsfier.cls.bias.data.detach().cpu().numpy()
    u = -np.matmul(np.linalg.pinv(w), b)

    with torch.no_grad():
        for i, (images, _) in enumerate(ood_loader):
            if device == None:
                images = Variable(images.cuda())
            else:
                images = Variable(images.to(device))
            nnOutputs = model(images)
            ood_feat = clsfier.intermediate_forward(nnOutputs).view(nnOutputs.shape[0], -1)
            # ood_feat = nnOutputs.view(nnOutputs.shape[0], -1)
            feature_ood.append(ood_feat.detach().cpu())
            nnOutputs = clsfier(nnOutputs)

            nnOutputs_np = to_np(nnOutputs.squeeze())
            logits_np = np.vstack((logits_np, nnOutputs_np))
        print("ood_feature")

        for i, (images, _) in enumerate(train_loader):
            if device == None:
                images = Variable(images.cuda())
            else:
                images = Variable(images.to(device))
            nnOutputs = model(images)
            train_feat = clsfier.intermediate_forward(nnOutputs).view(nnOutputs.shape[0], -1)
            # train_feat = nnOutputs.view(nnOutputs.shape[0], -1)
            feature_id_train.append(train_feat.detach().cpu())
            nnOutputs = clsfier(nnOutputs)

            nnOutputs_np = to_np(nnOutputs.squeeze())
            logits_np = np.vstack((logits_np, nnOutputs_np))
        print("train_feature")

        for i, (images, labels) in enumerate(val_loader):
            if device == None:
                images = Variable(images.cuda())
            else:
                images = Variable(images.to(device))
            nnOutputs = model(images)
            val_feat = clsfier.intermediate_forward(nnOutputs).view(nnOutputs.shape[0], -1)
            # val_feat = nnOutputs.view(nnOutputs.shape[0], -1)
            feature_id_val.append(val_feat.detach().cpu())
            nnOutputs = clsfier(nnOutputs)

            nnOutputs_np = to_np(nnOutputs.squeeze())
            logits_np = np.vstack((logits_np, nnOutputs_np))
            in_labels = np.vstack((in_labels, labels.cpu().numpy()))
        print("val_feature")

    feature_id_train = np.concatenate(feature_id_train, axis=0)
    feature_id_val= np.concatenate(feature_id_val, axis=0)
    feature_ood = np.concatenate(feature_ood, axis=0)
    # print(feature_ood.shape)
    # print(w.shape)
    # print(b.shape)

    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_oods = feature_ood @ w.T + b

    if feature_id_train.shape[-1] >= 2048:
        DIM = 1000
    elif feature_id_train.shape[-1] >= 768:
        DIM = 512
    else:
        DIM = feature_id_train.shape[-1] // 2
    print(f'{DIM=}')
    print('computing principal space...')

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print('computing alpha...')
    vlogit_id_train = np.linalg.norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')

    vlogit_id_val = np.linalg.norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = torch.logsumexp(torch.from_numpy(logit_id_val), axis=-1)
    print(vlogit_id_val.shape)
    print(energy_id_val.shape)
    score_id = -vlogit_id_val + energy_id_val.numpy()

    energy_ood = torch.logsumexp(torch.from_numpy(logit_oods), axis=-1)
    vlogit_ood = np.linalg.norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood.numpy()

    return score_id, score_ood # , to_np(in_labels)

def get_score(loader, model, clsfier, args, name=None, device=None):
    npy_file = args.save_path + str(args.score) + "_" + name + ".npy"
    label_npy_file = args.save_path + str(args.score) + "_" + "label.npy"
    
    print(npy_file, os.path.exists(npy_file))
    print(label_npy_file, os.path.exists(label_npy_file))
    
    if name == "in_test":
        head_position = [75]
    else:
        head_position = [0]
    df = pd.DataFrame(columns=['index', 'score'])
    for j in head_position:
        in_labels = np.empty([0, args.n_classes])
        label_npy_file = f'./score_result/{args.dataset}/label_argsort.npy'
        all_labels_argsort = np.load(label_npy_file)
        all_labels_argsort_head = all_labels_argsort[:j]
        # all_labels_argsort_head = all_labels_argsort[:-10]
        mask = torch.zeros((1, args.n_classes))
        mask[0, all_labels_argsort_head] = 1
        tail_nnOutputs_set = np.empty([0, args.n_classes])

        if not (os.path.exists(npy_file) and os.path.exists(label_npy_file)):
            logits_np = np.empty([0, args.n_classes])

            with torch.no_grad():
                for i, (images, labels) in enumerate(loader):
                    if device == None:
                        images = Variable(images.cuda())
                    else:
                        images = Variable(images.to(device))
                    nnOutputs = model(images)
                    nnOutputs = clsfier(nnOutputs)

                    nnOutputs_np = to_np(nnOutputs.squeeze())
                    logits_np = np.vstack((logits_np, nnOutputs_np))
                    if name == "in_test":
                        tail_index = (mask * labels).sum(1) == 0
                        tail_nnOutputs = nnOutputs_np[tail_index]
                        tail_nnOutputs_set = np.vstack((tail_nnOutputs_set, tail_nnOutputs))

            os.makedirs(args.save_path, exist_ok = True)

        else:
            logits_np = np.load(npy_file)
            # in_labels = np.load(label_npy_file)
            
        ## Compute the Score
        # if device == None:
        #     logits = torch.from_numpy(logits_np).cuda()
        #     in_labels = torch.from_numpy(in_labels).cuda()
        #     tail_nnOutputs_set = torch.from_numpy(tail_nnOutputs_set).cuda()
        # else:
        #     logits = torch.from_numpy(logits_np).to(device)
        #     in_labels = torch.from_numpy(in_labels).to(device)
        #     tail_nnOutputs_set = torch.from_numpy(tail_nnOutputs_set).to(device)

        logits = torch.from_numpy(logits_np)
        tail_nnOutputs_set = torch.from_numpy(tail_nnOutputs_set)
        E_f = torch.log(1 + torch.exp(logits))
        data = E_f.sum(1).cpu().numpy()
        
        if name == "in_test":
            E_f_tail = torch.log(1 + torch.exp(tail_nnOutputs_set))
            data_tail = E_f_tail.sum(1).cpu().numpy()

            # score_result_path = f'./score_distribution/{args.dataset}/{args.arch}/'
            # df_tail = pd.DataFrame(data)
            # df_tail.to_csv(score_result_path + f'75_{args.ood_data}_{args.score}.csv', index=False)

            new_types_tail = [j for _ in range(data_tail.shape[0])]
            new_data_tail = pd.DataFrame({'index': new_types_tail, 'score': data_tail})
            df = pd.concat([df, new_data_tail], ignore_index=True)

        new_types = [name] * len(data)
        new_data = pd.DataFrame({'index': new_types, 'score': data})

        df = pd.concat([df, new_data], ignore_index=True)
        # df = pd.DataFrame(data, columns=[args.method])
        score_result_path = f'./score_distribution/{args.dataset}/{args.arch}/'
        if not os.path.exists(score_result_path):
            os.makedirs(score_result_path, exist_ok=True)
        df.to_csv(score_result_path + f'{args.ood_data}_{args.ood}_{args.method}_{args.score}_{name}.csv', index=False)
    
    # score_result_path = f'./score_distribution/{args.dataset}/{args.arch}/'
    # if not os.path.exists(score_result_path):
    #     os.makedirs(score_result_path, exist_ok=True)
    # if name == "in_test":
    #     df.to_csv(score_result_path + f'{args.ood_data}_{args.ood}_{args.method}_{args.score}_id.csv', index=False)
    # else:
    #     df.to_csv(score_result_path + f'{args.ood_data}_{args.ood}_{args.method}_{args.score}_ood.csv', index=False)
    # if name == "in_test":
    #     columns = [name] + list(range(args.n_classes))
    #     label_inf = torch.cat([torch.from_numpy(scores).reshape(-1, 1), in_labels.cpu()], dim=1)
    #     logit_inf = torch.cat([torch.from_numpy(scores).reshape(-1, 1), logits.cpu()], dim=1)
    #     df_label = pd.DataFrame(label_inf.numpy(), columns=columns)
    #     df_logit = pd.DataFrame(logit_inf.numpy(), columns=columns)
    #     save_result = f'./score_result/{args.dataset}/{args.arch}'
    #     if not os.path.exists(save_result):
    #         os.makedirs(save_result)
    #     csv_label = f'./score_result/{args.dataset}/{args.arch}/label.csv'
    #     csv_logit = f'./score_result/{args.dataset}/{args.arch}/logit.csv'
    #     df_label.to_csv(csv_label, index=False)
    #     df_logit.to_csv(csv_logit, index=False)

    # return scores, to_np(in_labels)

def iterate_data_gradnorm(data_loader, model, clsfier, temperature, num_classes, args, name, device):
    confs = []
    in_labels = np.empty([0, args.n_classes])
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.to(device), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        outputs = clsfier(outputs)
        targets = torch.ones((inputs.shape[0], num_classes)).to(device)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        layer_grad = clsfier.module.cls.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

        if name == "in_test":
            in_labels = np.vstack((in_labels, y.cpu().numpy()))

    return np.array(confs), in_labels


def get_localoutlierfactor_scores(val, test, out_scores):
    import sklearn.neighbors
    scorer = sklearn.neighbors.LocalOutlierFactor(novelty=True)
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


def get_isolationforest_scores(val, test, out_scores):
    import sklearn.ensemble
    rng = np.random.RandomState(42)
    scorer = sklearn.ensemble.IsolationForest(random_state = rng)
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


class LinfPGDAttack:

    def __init__(
            self, model, eps=8.0, nb_iter=40,
            eps_iter=1.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, loss_func='CE', num_classes=10,
            elementwise_best=False):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.model = model
        self.num_classes = num_classes

        if loss_func == 'CE':
            # self.loss_func = nn.CrossEntropyLoss(reduction='none')
            self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        elif loss_func == 'OE':
            self.loss_func = OELoss()
        else:
            assert False, 'Not supported loss function {}'.format(loss_func)

        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone()
        if y is not None:
            y = y.detach().clone()
            y = y.cuda()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            outputs = self.model(x)
            loss = self.loss_func(outputs, y)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = torch.round(delta.data)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

        for ii in range(self.nb_iter):
            adv_x = x + delta / 255.0
            outputs = self.model(adv_x)

            if self.targeted:
                target = ((y + torch.randint(1, self.num_classes, y.shape).cuda()) % self.num_classes).long()
                loss = -self.loss_func(outputs, target)
            else:
                loss = self.loss_func(outputs, y)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta / 255.0
            outputs = self.model(adv_x)

            if self.targeted:
                target = ((y + torch.randint(1, self.num_classes, y.shape).cuda()) % self.num_classes).long()
                loss = -self.loss_func(outputs, target)
            else:
                loss = self.loss_func(outputs, y)

            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]

            adv_x = x + worst_perb / 255.0
        else:
            adv_x = x + delta.data / 255.0

        return adv_x
