import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d
from torch.autograd import Variable

# def focal_loss(input_values, gamma):
#     """Computes the focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values
#     return loss.mean()

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=0.):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, input, target):
#         return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)

class BCEWithThresholdLoss(nn.Module):  # STL-AUC-L2
    def __init__(self) -> None:
        super(BCEWithThresholdLoss, self).__init__()

    def forward(self, outputs, labels):
        outputs = torch.sigmoid(outputs)
        # print((labels * outputs).sum() / (labels.sum()))
        # print(((1 - labels) * outputs).sum() / ((1 - labels).sum()))

        # loss = labels * torch.log(F.relu(outputs - self.gamma) + 0.1) + (1 - labels) * outputs

        flattened_neg = ((1 - labels) * outputs).flatten()
        non_zeros_neg = flattened_neg[flattened_neg != 0]
        neg_top_k, _ = torch.topk(non_zeros_neg, k=int(1 * (1 - labels).sum()))

        flattened_pos = (labels * outputs).flatten()
        non_zeros_pos = flattened_pos[flattened_pos != 0]
        pos_bottom_k, _ = torch.topk(-non_zeros_pos, k=int(1 * labels.sum()))
        pos_bottom_k = -pos_bottom_k

        loss = F.relu(neg_top_k.mean() - pos_bottom_k.mean())
        print(loss)
        # print(neg_top_k.mean())
        # print(pos_bottom_k.mean())
        # if tail_outputs.shape[0] != 0:
        #     # print(torch.mean(tail_labels * torch.log(tail_outputs) + (1 - tail_labels) * torch.log(1 - tail_outputs)))
        #     loss = loss + torch.mean(tail_labels * torch.log(tail_outputs) + (1 - tail_labels) * torch.log(1 - tail_outputs))
        # else:
            
            # value_p, _ = torch.min(torch.log(1 + torch.exp(tail_outputs)) * tail_labels + (1 - tail_labels) * 10000, dim=0)
            # value_n, _ = torch.max(torch.log(1 + torch.exp(outputs)) * (1 - labels) - labels * 10000, dim=0)
            # print(value_n - value_p)
            # diff = torch.pow(F.relu(value_n - value_p), 2)
            # print(diff)
            # loss = diff.sum()

        return loss
    
class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


class AUROC(nn.Module):  # STL-AUC-L2
    def __init__(self, m) -> None:
        super(AUROC, self).__init__()
        # self.device = device
        self.m = m

    def forward(self, data_pos, data_neg):
        # pos = torch.FloatTensor(data_pos).to(self.device)
        # neg = torch.FloatTensor(data_neg).to(self.device)

        # pos = torch.sigmoid(data_pos)
        # neg = torch.sigmoid(data_neg)
        objective = (data_pos[:, None] - data_neg + self.m) ** 2
        
        return objective.mean()

class EnergyBCE(nn.Module):
    def __init__(self, m_p, m_n, device):
        super(EnergyBCE, self).__init__()
        self.m_p = 5
        self.m_n = -5
        self.device = device

    def forward(self, outputs, labels):
        '''
        outputs: n * c
        labels: n * c
        '''
        outputs = torch.sigmoid(outputs)
        Ec = torch.log(1 + torch.exp(outputs))
        # print('pos')
        # print((labels * Ec).sum() / (labels.sum()))
        # print('neg')
        # print(((1 - labels) * Ec).sum() / ((1 - labels).sum()))
        loss = torch.sum(labels * torch.log(outputs) * F.relu(self.m_p - Ec), dim=1) + torch.sum((1 - labels) * torch.log(1 - outputs) * F.relu(Ec - self.m_n), dim=1)
        return -loss.mean()
    
class EnergyBCE2(nn.Module):
    def __init__(self, m_p, m_n, beta, device):
        super(EnergyBCE2, self).__init__()
        self.m_p = 5
        self.m_n = -5
        self.beta = 0.01
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        '''
        outputs: n * c
        labels: n * c
        '''
        outputs = torch.sigmoid(outputs)
        Ec = torch.log(1 + torch.exp(outputs))
        # print('pos')
        # print((labels * Ec).sum() / (labels.sum()))
        # print('neg')
        # print(((1 - labels) * Ec).sum() / ((1 - labels).sum()))
        loss = self.criterion(outputs, labels)
        loss_p = F.relu(self.m_p - (labels * Ec).sum(1) / (labels.sum(1))).mean()
        loss_n = F.relu(((1 - labels) * Ec).sum(1) / ((1 - labels).sum(1)) - self.m_n).mean()
        loss = loss + self.beta * (loss_p + loss_n)
        return loss

class EnergyBCE3(nn.Module):
    def __init__(self, m, beta, device):
        super(EnergyBCE3, self).__init__()
        self.m = 2
        self.beta = 0.01
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        '''
        outputs: n * c
        labels: n * c
        '''
        outputs = torch.sigmoid(outputs)
        Ec = torch.log(1 + torch.exp(outputs))
        # print('pos')
        # print(torch.mean((labels * Ec).sum(1) / (labels.sum(1))))
        # print('neg')
        # print(torch.mean(((1 - labels) * Ec).sum(1) / ((1 - labels).sum(1))))
        loss = self.criterion(outputs, labels)
        gap = self.m + ((1 - labels) * Ec).sum(1) / ((1 - labels).sum(1)) - (labels * Ec).sum(1) / (labels.sum(1))
        loss_energy = torch.pow(F.relu(gap.mean()), 2)
        loss = loss + self.beta * loss_energy
        return loss
    
class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self.m_pos = 10
        self.m_neg = -10

    def forward(self, outputs, labels): 
        # print((F.relu(self.m_pos - labels * outputs)).mean())
        # print((F.relu((1 - labels) * outputs - self.m_neg)).mean())
        return (F.relu(self.m_pos - labels * outputs)).mean() + (F.relu((1 - labels) * outputs - self.m_neg)).mean()
        # return 1 - labels * torch.log(outputs) - (1 - labels) * torch.log(1 - outputs)

class ImbalanceBinaryCrossEntropy(nn.Module):
    def __init__(self, base_probs, device, margin=0.2, tau=1, use_margin=False):
        super(ImbalanceBinaryCrossEntropy, self).__init__()

        self.margin = margin
        self.tau = tau
        self.use_marging = use_margin
        self.device = device
        self.base_probs = torch.tensor(base_probs).to(self.device)

    def forward(self, x, y):
        '''
        x: logits n * c
        y: labels n * c
        base_probs: class prior 1 * c
        '''
        x_sigmoid = torch.sigmoid(x)
        # (self.base_probs**self.tau + 1e-12).view(1, -1)
        loss_pos = y * (torch.log(x_sigmoid))
        # loss_pos = y * (torch.log(torch.sigmoid(x)))
        if self.use_marging:
            hard_neg_position = torch.where((1 - y) * x_sigmoid <= self.margin, True, False)
            pos = torch.where((1 - y) * x_sigmoid > 0, True, False)
            x_sigmoid[hard_neg_position * pos] = 0
        loss_neg = (1 - y) * (torch.log(1 - x_sigmoid))

        loss = loss_pos + loss_neg
        # print((-y * (torch.log(torch.sigmoid(x)))).sum())
        # print((-loss_pos).sum())
        # print((-loss_neg).sum())

        return -loss.sum()

    
class FocalLoss(nn.Module):
    def __init__(self, gamma=4, eps=1e-8):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_w = torch.pow(1 - pt, self.gamma)

        loss *= one_sided_w

        return -loss.sum()
    

# def focal_loss(input_values, gamma):
#     """Computes the focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values
#     return loss.mean()
    
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=0.):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, input, target):
#         return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
    

class LDAMLoss(nn.Module):  # 改multi-label
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device=None):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # (1, c) (c, n)
        batch_m = batch_m.view((-1, 1)) 
        x_m = x - batch_m  # (n, c) - (n, 1) 每一个类都做减法
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class MultiLabelLDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device=None):
        super(MultiLabelLDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target):
        # index = torch.zeros_like(x, dtype=torch.uint8) 
        # index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = target.float().to(self.device)
        batch_m = torch.matmul(self.m_list[None, :].to(self.device), index_float.transpose(0, 1))  # (1, c) (c, n)
        batch_m = batch_m.view((-1, 1)) 
        x_m = x - batch_m  # (n, c) - (n, 1) 每一个类都做减法

        output = torch.where(target.bool(), x_m, x).to(self.device)
        return F.binary_cross_entropy_with_logits(self.s*output, target, weight=self.weight)
        # return F.cross_entropy(self.s*output, target, weight=self.weight)

class BinaryVSLoss(nn.Module):

    def __init__(self, iota_pos=0.0, iota_neg=0.0, Delta_pos=1.0, Delta_neg=1.0, weight=None):
        super(BinaryVSLoss, self).__init__()
        iota_list = torch.tensor([iota_neg, iota_pos]).to(torch.device('cuda'))
        Delta_list = torch.tensor([Delta_neg, Delta_pos]).to(torch.device('cuda'))

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros((x.shape[0], 2), dtype=torch.uint8)
        index_float = index.type(torch.cuda.FloatTensor)
        index_float.scatter_(1, target.long(), 1)

        batch_iota = torch.matmul(self.iota_list, index_float.t())
        batch_Delta = torch.matmul(self.Delta_list, index_float.t())

        batch_iota = batch_iota.view((-1, 1))
        batch_Delta = batch_Delta.view((-1, 1))

        output = x * batch_Delta - batch_iota

        return F.binary_cross_entropy_with_logits(30 * output, target, weight=self.weight)


class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None, device=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list).to(device)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list).to(device)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.binary_cross_entropy_with_logits(output, target, weight=self.weight)


def build_loss_fn(use_la_loss, base_probs, tau=1.0):
    def loss_fn(labels, logits):
        if use_la_loss:
            base_probs_tensor = torch.tensor(base_probs)  
            logits = logits + torch.log(base_probs_tensor**tau + 1e-12)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss.mean()
    return loss_fn

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.l_rate * epoch / 5
    elif epoch > 180:
        lr = args.l_rate * 0.0001
    elif epoch > 160:
        lr = args.l_rate * 0.01
    else:
        lr = args.l_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if args.rho_schedule == 'step':
      if epoch <= 5:
          rho = 0.05
      elif epoch > 180:
          rho = 0.6
      elif epoch > 160:
          rho = 0.5 
      else:
          rho = 0.1
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho
    if args.rho_schedule == 'linear':
      X = [1, args.epochs]
      Y = [args.min_rho, args.max_rho]
      y_interp = interp1d(X, Y)
      rho = y_interp(epoch)

      for param_group in optimizer.param_groups:

          param_group['rho'] = np.float16(rho)
    if args.rho_schedule == 'none':
      rho = args.rho
      for param_group in optimizer.param_groups:
          param_group['rho'] = rho

# for x, y in test_dataset:
#     logits = model(x, training=False)
#     test_acc_metric.update_state(y, logits)

#     if posthoc_adjusting:
#         # Posthoc logit-adjustment.
#         adjusted_logits = logits - tf.math.log(tf.cast(base_probs**FLAGS.tau + 1e-12, dtype=tf.float32))
#         test_adj_acc_metric.update_state(y, adjusted_logits)