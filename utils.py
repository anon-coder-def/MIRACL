from __future__ import absolute_import
from __future__ import print_function

from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, cohen_kappa_score, mean_squared_error, f1_score, average_precision_score
from sklearn import metrics
import torch

import numpy as np
import platform
import pickle
import json
import os


def is_ascending(lst):
    for i in range(1, len(lst)):
        if lst[i-1] > lst[i]:
            return 0  # 如果存在非升序的情况，返回0
    return 1  # 列表是升序的情况下返回1



class Discretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path= 'normalizers/discretizer_config.json'):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]           # Hours列
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)          # mask值为1代表该处是真实值，非填充值
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:               # 没有历史数据时填充常规值（json文件中）
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret


def my_metrics(yt, yp, task=None):
    if task in ['in-hospital-mortality', 'decompensation', 'readmission']:
        yt = yt.view(-1).detach().cpu().numpy()
        yp = yp.view(-1).detach().cpu().numpy()
        precision, recall, _, = precision_recall_curve(yt, yp)
        aupr = metrics.auc(recall, precision)
        auc = roc_auc_score(yt, yp)
        return auc, aupr
    elif task == 'phenotyping' or task == 'diagnosis':
        yt = yt.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        total_auc = 0.
        for i in range(yt.shape[1]):
            label_mask = (yt[:, i] > -1)
            try:
                auc = roc_auc_score(yt[:, i][label_mask], yp[:, i][label_mask])
            except ValueError:
                auc = 0.5
            total_auc += auc
        macro_auc = total_auc/yt.shape[1]

        label_mask = (yt > -1)
        try:
            micro_auc = roc_auc_score(yt[label_mask], yp[label_mask])
        except ValueError:
            micro_auc = 0.5
        return macro_auc, micro_auc
    else:
        micro_F1 = f1_score(yt.detach().cpu().numpy(), yp.detach().cpu().numpy(), average="micro")
        macro_F1 = f1_score(yt.detach().cpu().numpy(), yp.detach().cpu().numpy(), average="macro")
        return micro_F1, macro_F1

    #---f1,acc,recall, specificity, precision
    # f1, thresholds = f1_score_binary(yt, yp)
    # acc = accuracy_binary(yt, yp, thresholds)
    # precision = precision_binary(yt, yp, thresholds)
    # recall = recall_binary(yt, yp, thresholds)
    # mcc = mcc_binary(yt, yp, thresholds)

    # return auc, aupr, f1_score[0, 0], accuracy[0, 0] #, recall[0, 0], specificity[0, 0], precision[0, 0]



def robust_metrics(yt, yp, task=None):
    """
    Compute Mean Average Precision (mAP), overall F1, and per-class F1 for multi-label classification.
    
    Args:
        yt (torch.Tensor): Ground truth binary labels.
        yp (torch.Tensor): Predicted probabilities or scores.
        task (str): Task name, if specific processing is required.
    
    Returns:
        dict: Dictionary containing mAP, overall F1, and per-class F1 scores.
    """
    # Convert tensors to NumPy arrays
    yt = yt.detach().cpu().numpy()
    yp = yp.detach().cpu().numpy()

    # Binary predictions for F1 score calculation
    yt_binary = (yt > 0.5).astype(int)
    yp_binary = (yp > 0.5).astype(int)

    # Initialize metrics
    metrics = {}

    # Compute Mean Average Precision (mAP)
    total_ap = 0.0
    for i in range(yt.shape[1]):
        label_mask = (yt[:, i] > -1)  # Ignore invalid labels
        try:
            ap = average_precision_score(yt[:, i][label_mask], yp[:, i][label_mask])
        except ValueError:
            ap = 0.0
        total_ap += ap
    metrics['mAP'] = total_ap / yt.shape[1]

    # Compute Overall F1
    
    
    metrics['Overall F1'] = f1_score(yt_binary.flatten(), yp_binary.flatten())

    # Compute Per-class F1
    per_class_f1 = []
    for i in range(yt_binary.shape[1]):
        try:
            f1 = f1_score(yt_binary[:, i], yp_binary[:, i])
        except ValueError:
            f1 = 0.0
        per_class_f1.append(f1)
    metrics['Per-class F1'] = sum(per_class_f1) / len(per_class_f1) 

    return metrics['mAP'], metrics['Overall F1'], metrics['Per-class F1']



# making mask for input sequence, in which 0 for real and 1 for unreal
def length_to_mask(length, max_len=None, dtype=None):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    mask = ~mask
    return mask



import torch
import torch.nn.functional as F


class BinaryFocalLoss(torch.nn.Module):
    """
    Binary Focal Loss using PyTorch built-in commands.
    Args:
        gamma (float): Focusing parameter to down-weight easy examples. Default is 2.0.
        alpha (float or None): Weight for positive examples. Default is None (no weighting).
        reduction (str): 'none' | 'mean' | 'sum'. Specifies the reduction to apply. Default is 'mean'.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Forward pass for Binary Focal Loss.
        Args:
            logits (torch.Tensor): Raw output logits of shape (batch_size, num_labels).
            labels (torch.Tensor): Ground truth binary labels of shape (batch_size, num_labels).
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(probs, labels, reduction='none')

        # Apply the focal loss formula
        focal_weight = torch.pow(1 - probs, self.gamma) * labels + torch.pow(probs, self.gamma) * (1 - labels)
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_loss *= alpha_weight

        # Reduce loss based on the specified reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
import torch.nn as nn    


import torch.nn.functional as F
def bpr_loss(pos_score, neg_score):
    return -torch.mean(F.logsigmoid(pos_score - neg_score))



class MultiLabelRankingLoss(nn.Module):
    """
    Multi-label Ranking Loss for Multi-label Classification.
    Encourages scores for positive labels to be higher than negative labels.
    """
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Args:
            margin (float): Margin for the ranking loss.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(MultiLabelRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, probs, labels):
        """
        Args:
            logits (torch.Tensor): Predicted logits (N x C).
            labels (torch.Tensor): Binary ground truth labels (N x C).

        Returns:
            torch.Tensor: Computed ranking loss.
        """
        # Convert logits to probabilities if not already

        # Expand dimensions to compute pairwise differences
        pos_scores = probs.unsqueeze(2)  # N x C x 1
        neg_scores = probs.unsqueeze(1)  # N x 1 x C

        # Mask for valid positive and negative pairs
        pos_mask = labels > 0  # N x C
        neg_mask = labels == 0  # N x C
        valid_pairs = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)  # N x C x C

        # Compute pairwise differences
        pairwise_diff = self.margin - (pos_scores - neg_scores)  # N x C x C

        # Apply ReLU to keep only violations of the margin
        ranking_loss = F.relu(pairwise_diff)

        # Mask invalid pairs
        masked_loss = ranking_loss * valid_pairs  # N x C x C

        # Apply reduction
        if self.reduction == 'none':
            return masked_loss  # Return all individual losses without reduction
        else:
            num_pairs = valid_pairs.sum().float() + 1e-8  # Avoid division by zero
            if self.reduction == 'mean':
                return masked_loss.sum() / num_pairs
            elif self.reduction == 'sum':
                return masked_loss.sum()
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
    
    
import torch
import torch.nn as nn

class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross-Entropy Loss for robust multi-label classification.
    Combines forward cross-entropy loss and reverse KL-divergence loss.
    """
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-8, reduction='mean'):
        """
        Args:
            alpha (float): Weight for forward Cross-Entropy loss.
            beta (float): Weight for reverse KL-divergence loss.
            epsilon (float): Small constant for numerical stability.
            reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'.
        """
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model logits (N x C).
            targets (torch.Tensor): Binary ground truth labels (N x C).

        Returns:
            torch.Tensor: Symmetric Cross-Entropy loss.
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Clamp targets for numerical stability
        targets = targets.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        # Forward Cross-Entropy Loss
        ce_loss = - (targets * torch.log(probs + self.epsilon) + (1 - targets) * torch.log(1 - probs + self.epsilon))
        
        # Reverse KL-Divergence Loss
        reverse_kl = - (probs * torch.log(targets + self.epsilon) + (1 - probs) * torch.log(1 - targets + self.epsilon))
        
        # Combine losses
        combined_loss = self.alpha * ce_loss + self.beta * reverse_kl

        # Apply reduction
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        elif self.reduction == 'none':
            return combined_loss  # Element-wise loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Use 'none', 'mean', or 'sum'.")
        
   

   
   
        