import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CELoss(nn.Module):
    def __init__(self, weight, scale=1):
        super(CELoss, self).__init__()
        self.weight = weight
        self.s = scale

    def forward(self, logits, targets):
        return F.cross_entropy(self.s * logits, targets, reduction='none', weight=self.weight)

# LDAM and Focal loss from https://github.com/kaidic/LDAM-DRW
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss for Long-Tailed Learning.
    Reference: https://arxiv.org/abs/2004.00888
    """
    def __init__(self, class_counts, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        # class_counts should be tensor of shape [num_classes]
        self.register_buffer('class_counts', class_counts.float())
        self.reduction = reduction

    def forward(self, logits, labels):
        # logits: [batch_size, num_classes]
        # labels: [batch_size]

        # Add log prior: log(1/n_i) = -log(n_i)
        log_priors = torch.log(self.class_counts + 1e-12)
        logits_adjusted = logits - log_priors  # subtract log class prior

        loss = F.cross_entropy(logits_adjusted, labels, reduction=self.reduction)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0., scale=1):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.s = scale

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input * self.s, target, reduction='none', weight=self.weight), self.gamma)


# LDAM and Focal loss from https://github.com/kaidic/LDAM-DRW
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), True)
        
        index_float = index.type(x.dtype).to(x.device)
        m_list = self.m_list.to(x.device)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
        #batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        weight = self.weight.to(x.device) if self.weight is not None else None
        return F.cross_entropy(self.s * output, target, weight=self.weight, reduction='none')
    
class CBWLoss(nn.Module):
    def __init__(self, freq, reduction='mean'):
        super(CBWLoss, self).__init__()
        freq = freq.clone()
        freq[freq == 0] = 1  # prevent div-by-zero
        weight = freq.sum() / (freq.shape[0] * freq)
        weight = weight.type(torch.float32)
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weight, reduction=self.reduction)
    
 
class GRWLoss(nn.Module):
    def __init__(self, freq, exp_scale=1.2, reduction='mean'):
        super(GRWLoss, self).__init__()
        freq = freq.clone()
        freq[freq == 0] = 1  # prevent div-by-zero
        num_classes = freq.shape[0]
        exp_reweight = 1 / (freq ** exp_scale)
        exp_reweight = exp_reweight / exp_reweight.sum() * num_classes
        exp_reweight = exp_reweight.type(torch.float32)
        self.register_buffer('weight', exp_reweight)
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weight, reduction=self.reduction)   
    