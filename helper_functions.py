import os
from copy import deepcopy
import random
import time
import math

import numpy as np
from numpy import random as nr
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import label_ranking_loss

def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def average(self):
        return self.avg
    
    def value(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)


    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.clone().cpu().numpy()
        n, c = self.scores.size()
        scores = np.zeros((n, c))
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0.5 else 0
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            Ng[k] = np.sum(targets == 1)  # TP + FN 所有真实标签为1的数量
            Np[k] = np.sum(scores >= 0.5)  # TP + FP 所有预测为1的数量
            Nc[k] = np.sum(targets * (scores >= 0.5))  # TP 所有真实标签为1 且 预测为1的数量
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)  # TP / (TP + FP)
        OR = np.sum(Nc) / np.sum(Ng)  # TP / (TP + FN)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class  # TP[k] / (TP[k] + FP[k]) / num_class
        CR = np.sum(Nc / Ng) / n_class  # TP[k] / (TP[k] + FN[k]) / num_class 
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

    def ranking_loss(self):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        return label_ranking_loss(targets, scores)

    def precision_at_k(self, k):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        top_k_idx = np.argpartition(scores, kth=-k)[:, -k:]
        top_k_targets = targets[np.arange(targets.shape[0])[:, np.newaxis], top_k_idx]
        top_k_scores = np.ones((targets.shape[0], k))

        return precision_score(top_k_targets, top_k_scores, average='samples')

    def recall_at_k(self, k):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()
        top_k_idx = np.argpartition(scores, kth=-k)

        top_k_scores = np.zeros((scores.shape[0], scores.shape[1]))
        top_k_scores[np.arange(targets.shape[0])[:, np.newaxis], top_k_idx[:, -k:]] = 1

        return recall_score(targets, top_k_scores, average='samples')

    def precision_at_k_instance(self, targets, scores, top_k_scores, k):

        top_k_idx = np.argpartition(scores, kth=-k)[-k:]
        top_k_targets = targets[top_k_idx]
        # top_k_scores = np.ones(k)
        return precision_score(top_k_targets, top_k_scores[:k], average='micro')

    def mAP_at_K(self, K):
        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        top_k_idx = np.argsort(scores)[:, ::-1][:, :K]
        top_k_scores = np.ones(K)
        top_k_targets = targets[np.arange(targets.shape[0])[:, np.newaxis], top_k_idx]
        # print(top_k_scores)
        # print(top_k_targets)
        p_at_k = np.array([[self.precision_at_k_instance(t, p, top_k_scores, k) for k in range(1, K + 1)] for t, p in zip(targets, scores)])
        # print(p_at_k)

        # scores = torch.tensor([[0.98, 0.45, 0.85, 0.8], [0.48, 0.95, 0.87, 0.8], [0.1, 0.4, 0.35, 0.8]])
        # targets = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0]])

        n_k = np.sum(targets, axis=1)
        tmp_k = np.ones(targets.shape[0]) * K
        tag = n_k <= tmp_k
        n_k[~tag] = K

        # print(p_at_k * top_k_targets)
        # print(np.sum(p_at_k * top_k_targets, axis=1) / n_k)

        return np.average(np.sum(p_at_k * top_k_targets, axis=1) / n_k)

    def NDCG_at_K(self, K):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()
        return ndcg_score(y_true=targets, y_score=scores, k=K)

    def _tie_averaged_dcg(self, y_true, y_score, discount_cumsum):
        _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
        ranked = np.zeros(len(counts))
        np.add.at(ranked, inv, y_true)
        ranked /= counts
        groups = np.cumsum(counts) - 1
        discount_sums = np.empty(len(counts))
        discount_sums[0] = discount_cumsum[groups[0]]
        discount_sums[1:] = np.diff(discount_cumsum[groups])
        return (ranked * discount_sums).sum()

    def DCG_l_at_K(self, targets, scores, K, ignore_ties=False):
        # discount1 = 1 / (np.log(np.arange(targets.shape[1]) + 2) / np.log(2))
        # discount2 = 1 / (np.log(np.arange(targets.shape[1]) + 2) / np.log(2))
        discount = K - np.arange(targets.shape[1])

        if K is not None:
            discount[K:] = 0
        
        if ignore_ties:
            ranking = np.argsort(scores)[:, ::-1]
            ranked = targets[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
            cumulative_gains = discount.dot(ranked.T)

            # cumulative_gains = n_k * (n_k + 1) / 2
            # print(cumulative_gains.shape)
        else:
            discount_cumsum = np.cumsum(discount)
            cumulative_gains = [
                self._tie_averaged_dcg(y_t, y_s, discount_cumsum)
                for y_t, y_s in zip(targets, scores)
            ]
            cumulative_gains = np.asarray(cumulative_gains)

        return cumulative_gains

    def NDCG_l_at_K(self, K):
        targets = self.targets.clone().cpu().numpy()  # 0-1 array
        scores = self.scores.clone().cpu().numpy()
        gain = self.DCG_l_at_K(targets, scores, K, ignore_ties=False)
        normalizing_gain = self.DCG_l_at_K(targets, targets, K, ignore_ties=True)
        all_irrelevant = normalizing_gain == 0
        gain[all_irrelevant] = 0
        gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]

        return np.average(gain)

    def DCG_ln_at_K(self, K):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        n_k = np.sum(targets, axis=1)
        tmp_k = np.ones(targets.shape[0]) * K
        tag = n_k <= tmp_k
        n_k[~tag] = K
        gain = self.DCG_l_at_K(targets, scores, K) / n_k

        return np.average(gain)

    def AUTKC_M(self, K):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()
        gain = self.DCG_l_at_K(targets, scores, K) / K

        return np.average(gain)

    def AUTKC_L(self, K):

        return self.DCG_ln_at_K(K) / K

    def AUTKC_Q(self, K):

        return self.NDCG_l_at_K(K) / K


from pycocotools.coco import COCO

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, missing=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.missing = missing

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.missing == 1:
            pos = torch.nonzero(target).squeeze(1)
            size = pos.shape[0] * 0.5 // 1 + 1
            choose = nr.choice(pos, size=size, replace=False)
            target[choose] = 0
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def cal_confounder(train_loader,model,args):
    model = model.eval()
    if 'res' in args.backbone:
        feature_map_size = args.image_size // 32
        cfer = torch.zeros((args.num_classes,args.feat_dim,feature_map_size,feature_map_size)).cuda()
    else:
        cfer = torch.zeros((args.num_classes,args.feat_dim)).cuda()
    
    num_classes = torch.zeros((args.num_classes)).cuda()
    
    for i, (inputData, target) in enumerate(train_loader): 
        inputData = inputData.cuda()
        target = target.cuda()  
       
        feat,logits = model(inputData)
       
        target_nz = target.nonzero()
        for label_info in target_nz:
            batch_id, label = label_info
            batch_id, label = int(batch_id), int(label)
            cfer[label] += feat.data[batch_id]
            num_classes[label] += 1
    if cfer.dim() > 2:
        cfer = cfer.flatten(2).mean(dim=-1)
    cfer = cfer / num_classes[:,None]
    model = model.train()
    return cfer

def model_transfer(model,tde_model,confounder,args):
    model = model.module
    state = model.state_dict()

    if not args.use_intervention:
        filtered_dict = {k: v for k, v in state.items() if (k in model.state_dict() and 'clf.logit_clf' not in k)}
        tde_model.load_state_dict(filtered_dict,strict=False)
    else:
        tde_model.load_state_dict(state)
    
    tde_model.clf.stagetwo = True
    tde_model.clf.memory.data = confounder

    tde_model = nn.DataParallel(tde_model).cuda()
    model = model.cpu()
    del model

    return tde_model

# import tqdm

# def aysmmetric_pseudo_labeling(model, P, Z, epoch, train_loader, device):

#     model.eval()

#     total_preds = None
#     total_idx = None
#     steps_per_epoch = len(train_loader)

#     desc = '[{}/{}]{}'.format(epoch, 100, 'PL'.rjust(8, ' '))
#     for i, (image, target) in enumerate(tqdm(train_loader, desc=desc)):

#         batch = i

#         # move data to GPU:
#         image = image.to(device, non_blocking=True)

#         # forward pass:
#         with torch.set_grad_enabled(False):
#             logits = model.fc(image)
#             preds = torch.sigmoid(logits)
#             if preds.dim() == 1:
#                 preds = torch.unsqueeze(preds, 0)

#         # gather:
#         if batch == 0:
#             total_preds = preds.detach().cpu().numpy()
#             total_idx = i.cpu().numpy()
#         else:
#             total_preds = np.vstack((preds.detach().cpu().numpy(), total_preds))
#             total_idx = np.hstack((i.cpu().numpy(), total_idx))

#             # pseudo-label:
#             if batch >= steps_per_epoch - 1:
#                 for i in range(total_preds.shape[1]):  # class-wise
#                     class_preds = total_preds[:, i]
#                     class_labels_obs = train_loader.label_matrix_obs[:, i]
#                     class_labels_obs = class_labels_obs[total_idx]

#                     # select unlabel data:
#                     unlabel_class_preds = class_preds[class_labels_obs == 0]
#                     unlabel_class_idx = total_idx[class_labels_obs == 0]

#                     # select samples:
#                     neg_PL_num = int(0.9 * P['unlabel_num'][i] / (P['num_epochs'] - 5))
#                     sorted_idx_loc = np.argsort(unlabel_class_preds)  # ascending
#                     selected_idx_loc = sorted_idx_loc[:neg_PL_num]  # select indices

#                     # assgin soft labels:
#                     for loc in selected_idx_loc:
#                         Z['datasets']['train'].label_matrix_obs[unlabel_class_idx[loc], i] = -unlabel_class_preds[loc]


if __name__ == "__main__":
    # scores = np.array([[0.8, 0.7, 0.1, 0.75], [0.2, 0.5, 0.6, 0.9]])
    # target = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    
    scores = torch.tensor([[0.98, 0.45, 0.85, 0.8]])
    targets = torch.tensor([[1, 0, 0, 0]])
    a = AveragePrecisionMeter()
    a.scores = scores
    a.targets = targets
    # print(a.precision_at_k(3))
    # print(a.recall_at_k(3))
    # print(a.NDCG_at_K(3))
    # print(a.NDCG_l_at_K(3))
    print(a.mAP_at_K(3))
    # print(a.AUTKC_L(3))
    # print(a.AUTKC_M(3))
    # print(a.AUTKC_Q(3))
