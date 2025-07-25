import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mimic4noisy.gaussian_model import *
from utils import *
from baseline.LossFunction.Loss import *
from mimic4noisy.beta_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
def standardization_minmax(data):
    mini = np.min(data, axis=0)
    maxi = np.max(data, axis=0)
    return (data - mini) / (maxi-mini)



def normalize(data):
    """
    Normalize data to range [0, 1] for both numpy arrays and torch tensors.
    
    Args:
        data (np.ndarray or torch.Tensor): Input data.
    Returns:
        np.ndarray or torch.Tensor: Normalized data.
    """
    if isinstance(data, np.ndarray):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
    elif isinstance(data, torch.Tensor):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 1e-9)
    else:
        raise TypeError("Unsupported data type. Use numpy.ndarray or torch.Tensor.")



def select_class_by_class(model_loss,loss_all,pred_all,args,epoch,x_idxs,labels, task_index):
    '''
        single class confident sample selection EPS
    '''
    gamma = 0.5

    if args.mean_loss_len > epoch:
        loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
            loss_all[x_idxs, :epoch].mean(axis=1))
    else:
        if args.mean_loss_len < 2:
            loss_mean = loss_all[x_idxs, epoch]
        else:
            loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
                loss_all[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

    # STANDARDIZE LOSS FOR EACH CLASS
    labels_numpy = labels.detach().cpu().numpy().squeeze()
    recreate_idx=torch.tensor([]).long()
    batch_idxs = torch.tensor(np.arange(len(model_loss))).long()
    standar_loss = np.array([])
    
    
    for i in range(args.nbins[torch.unique(task_index)]):
        if (labels_numpy==i).sum()>1:
            if args.standardization_choice == 'z-score':
                each_label_loss = standardization(loss_mean[labels_numpy==i])
            else:
                each_label_loss = standardization_minmax(loss_mean[labels_numpy == i])
                
                
            standar_loss = np.concatenate((standar_loss,each_label_loss))
            recreate_idx=torch.cat((recreate_idx,batch_idxs[labels_numpy==i]))
        elif (labels_numpy==i).sum()==1:
            standar_loss = np.concatenate((standar_loss, [0.]))
            recreate_idx=torch.cat((recreate_idx,batch_idxs[labels_numpy==i]))

    # SELECT CONFIDENT SAMPLES
    
    _, model_sm_idx = torch.topk(torch.from_numpy(standar_loss), k=int(standar_loss.size*(standar_loss<=standar_loss.mean()).mean()), largest=False)

    model_sm_idxs = recreate_idx[model_sm_idx]
    

    # SELECT LESS CONFIDENT SAMPLES 
    _, less_confident_idx = torch.topk(torch.from_numpy(standar_loss), k=int(standar_loss.size * (standar_loss > standar_loss.mean()).mean()), largest=True)
    less_confident_idxs = recreate_idx[less_confident_idx]

    # CALCULATING L_CONF
    model_loss_filter = torch.zeros((model_loss.size(0))).to(device)
    model_loss_filter[model_sm_idxs] = 1.0
    L_conf = (model_loss_filter * model_loss).mean()

    return L_conf, model_sm_idxs, less_confident_idxs


def calculate_label_similarity(labels):
    """
    Calculate label similarity matrix based on co-occurrence.
    
    Args:
        labels (np.ndarray): Multi-hot label matrix, shape (n_samples, n_labels).
    
    Returns:
        np.ndarray: Label similarity matrix, shape (n_labels, n_labels).
    """
    co_occurrence = np.dot(labels.T, labels)  # Co-occurrence matrix
    frequency = np.sum(labels, axis=0)  # Label frequencies
    similarity = co_occurrence / (np.sqrt(np.outer(frequency, frequency)) + 1e-9)
    return similarity


def calculate_diversity(probabilities, labels, alpha=1.0, beta=1.0, gamma=0.5):
    """
    Calculate selection metric combining uncertainty, rank, and contextual diversity.
    
    Args:
        probabilities (np.ndarray): Predicted probabilities, shape (n_samples, n_labels).
        loss (np.ndarray): Loss values for each instance-label pair, shape (n_samples, n_labels).
        labels (np.ndarray): Ground truth labels, shape (n_samples, n_labels).
        alpha (float): Weight for uncertainty.
        beta (float): Weight for rank.
        gamma (float): Weight for contextual diversity.
    
    Returns:
        np.ndarray: Selection metric scores, shape (n_samples, n_labels).
    """
    # Calculate uncertainty (entropy)
    epsilon = 1e-9
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    # Calculate contextual diversity
    similarity_matrix = calculate_label_similarity(labels)
    diversity = 1 - np.dot(probabilities, similarity_matrix) / (np.sum(probabilities, axis=1, keepdims=True) + epsilon)

    # Combine metrics
    selection_metric = diversity
    return selection_metric


def select_memorization_and_forgetting_per_epoch(pred_all, labels, clean_labels, epoch, lambda_coeff=1.0):

    batch_size, n_bins = labels.shape
    
    # Initialize memorization and forgetting difficulty
    memorization = torch.zeros((batch_size, n_bins), device = device)
    forgetting = torch.zeros((batch_size, n_bins), device = device)
    
    curr_pred_all = pred_all[:, :, :epoch]

    for i in range(batch_size):  # Iterate over each sample
        for j in range(n_bins):  # Iterate over each label (bin)
            # Extract prediction sequence for sample and label
            pred_sequence = [int(round(pred)) for pred in curr_pred_all[i, j, :].tolist()]  # Convert predictions to integers (0 or 1)
            true_label = int(labels[i, j])

            # Convert prediction sequence to binary (1 if correct, 0 otherwise)
            binary_sequence = [1 if pred == true_label else 0 for pred in pred_sequence]


            # Identify segments of memorized (1) and misclassified (0)
            segments = []
            current_segment = [binary_sequence[0]]
            for pred in binary_sequence[1:]:
                if pred == current_segment[-1]:
                    current_segment.append(pred)
                else:
                    segments.append(current_segment)
                    current_segment = [pred]
            segments.append(current_segment)

            # Calculate memorization and forgetting difficulties
            mem_segments = [seg for seg in segments if seg[0] == 0]  # Misclassified segments
            forget_segments = [seg for seg in segments if seg[0] == 1]  # Memorized segments
            
            if len(mem_segments) > 0:
                memorization[i, j] = sum(len(seg) for seg in mem_segments) / len(mem_segments)
            if len(forget_segments) > 0:
                forgetting[i, j] = sum(len(seg) for seg in forget_segments) / len(forget_segments)

    # Calculate selection metric
    
    selection_metric = memorization - lambda_coeff * forgetting 

    return memorization, forgetting, torch.tensor(selection_metric, device = device)
    





def select_memorization_and_forgetting(pred_all, labels, clean_labels, x_ids, epoch, lambda_coeff=1.0):
    """
    Calculate memorization (M), forgetting (F), and selection metric (C) for multi-label data.
    
    Args:
        pred_all (torch.Tensor): Historical predictions, shape (n_samples, n_bins, n_epochs).
        labels (torch.Tensor): True labels, shape (n_samples, n_bins).
        lambda_coeff (float): Coefficient for weighting forgetting in the selection metric.
    
    Returns:
        memorization (torch.Tensor): Memorization difficulty, shape (n_samples, n_bins).
        forgetting (torch.Tensor): Forgetting difficulty, shape (n_samples, n_bins).
        selection_metric (torch.Tensor): Combined metric, shape (n_samples, n_bins).
    """
    
    # Stage 1:  Select top k indices for each label (1-25) for both 1 and 0 class    
    labels_numpy = labels.detach().cpu().numpy().squeeze() 

    batch_size, n_bins = labels_numpy.shape
    
    # Initialize memorization and forgetting difficulty
    memorization = torch.zeros((batch_size, n_bins), device = device)
    forgetting = torch.zeros((batch_size, n_bins), device = device)
    
    curr_pred_all = pred_all[x_ids, :, :epoch]
    
    
    for i in range(batch_size):  # Iterate over each sample
        for j in range(n_bins):  # Iterate over each label (bin)
            # Extract prediction sequence for sample and label
            pred_sequence = [int(round(pred)) for pred in curr_pred_all[i, j, :].tolist()]  # Convert predictions to integers (0 or 1)
            true_label = int(labels_numpy[i, j])

            # Convert prediction sequence to binary (1 if correct, 0 otherwise)
            binary_sequence = [1 if pred == true_label else 0 for pred in pred_sequence]


            # Identify segments of memorized (1) and misclassified (0)
            segments = []
            current_segment = [binary_sequence[0]]
            for pred in binary_sequence[1:]:
                if pred == current_segment[-1]:
                    current_segment.append(pred)
                else:
                    segments.append(current_segment)
                    current_segment = [pred]
            segments.append(current_segment)

            # Calculate memorization and forgetting difficulties
            mem_segments = [seg for seg in segments if seg[0] == 0]  # Misclassified segments
            forget_segments = [seg for seg in segments if seg[0] == 1]  # Memorized segments
            
            if len(mem_segments) > 0:
                memorization[i, j] = sum(len(seg) for seg in mem_segments) / len(mem_segments)
            if len(forget_segments) > 0:
                forgetting[i, j] = sum(len(seg) for seg in forget_segments) / len(forget_segments)

    # Calculate selection metric
    selection_metric = memorization - lambda_coeff * forgetting 
    
    return memorization, forgetting, torch.tensor(selection_metric, device = device)





def fit_gaussian_model_loss_corr_perclassepoch(args, model, loss, rank, C, divers, criterion_now_each, labels, pred_labels, true_labels, epoch):
    """
    Perform correction for noisy labels using GMM fitting for positive pairs per class and  for negative pairs.

    Args:
        loss (torch.Tensor): Per-sample loss (shape: [N, D]).
        rank (torch.Tensor): Per-sample rank (shape: [N, D]).
        C (torch.Tensor): Correlation matrix (shape: [D, D]).
        criterion_now_each (function): Loss function to compute corrections.
        labels (torch.Tensor): Binary noisy labels (shape: [N, D]).
        pred_labels (torch.Tensor): Predicted probabilities (shape: [N, D]).
        true_labels (torch.Tensor): Ground truth labels (shape: [N, D]).

    Returns:
        torch.Tensor: Corrected loss.
    """
    # Normalize the selection metric
    

    pos_selection_metric = args.mem_coef * normalize(C) + args.rank_coef * normalize(rank)
    neg_selection_metric =  args.mem_coef * normalize(C) + args.rank_coef * normalize(rank)

           
    original_labels = labels.clone()
        
    
    
    threshold = args.threshold_coef
    # Control EMA coefficient
    batch_size, num_classes = labels.shape

    # Compute probabilities for negative labels
    Corr = get_correlation_matrix(labels)
    negative_corr = labels @ Corr
    negative_pair_mask = (negative_corr > threshold) & (labels == 0)
    

    # # Initialize clean mask
    clean_mask = torch.zeros_like(labels, dtype=torch.float32)
    soft_correction_weight = 0.5  # Adjust this weight as needed for soft correction

    # Iterate over each class for positive GMM fitting
    for class_idx in range(num_classes):
        class_pos_indices = labels[:, class_idx] > 0.5
        class_neg_indices = labels[:, class_idx] <= 0.5

        # Correct Correlated Negative Labels ONLY
        class_neg_pair_indices = negative_pair_mask[:, class_idx] == 1
        
        # Selection Metric For Positive Pair and Negative Pair Separately
        class_selection_metric = pos_selection_metric[class_pos_indices, class_idx]
        neg_class_selection_metric = neg_selection_metric[class_neg_indices, class_idx]

        if len(class_selection_metric) > 1:
            # Fit GMM for positive samples in this class
            class_selection_metric_np = class_selection_metric.detach().cpu().numpy()
            pos_gmm = fit_gaussian_model(class_selection_metric_np)
            pos_clean_mean, pos_noisy_mean = np.sort(pos_gmm.means_.flatten())
            
            ############################## POSITIVE CORRECTION #############################
            pos_clean_indices = class_pos_indices & (pos_selection_metric[:, class_idx] <= pos_clean_mean)
            pos_uncertain_indices = class_pos_indices & (pos_clean_mean < pos_selection_metric[:, class_idx]) & (pos_selection_metric[:, class_idx] <= pos_noisy_mean)
            pos_noisy_indices = class_pos_indices & (pos_selection_metric[:, class_idx] > pos_noisy_mean)
            
            # Update clean mask and store clean indices
            clean_mask[pos_clean_indices, class_idx] = 1  # Mark positives as clean

            # Hard Correction
            # labels[pos_clean_indices, class_idx] = 1
            # labels[pos_noisy_indices, class_idx] = 0  # Hard correction for noisy samples
            
            # For clean label, use original label
            # For noisy label, use prediction
            labels[pos_noisy_indices, class_idx] = pred_labels[pos_noisy_indices, class_idx]
            
            # Soft correction for uncertain samples
            labels[pos_uncertain_indices, class_idx] = (
                soft_correction_weight * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - soft_correction_weight) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)

        if len(neg_class_selection_metric) > 1:
            # Fit GMM for negative samples in this class
            neg_class_sm_np = neg_class_selection_metric.detach().cpu().numpy()
            neg_gmm = fit_gaussian_model(neg_class_sm_np)
            neg_clean_mean, neg_noisy_mean = np.sort(neg_gmm.means_.flatten())

            ############################## NEGATIVE CORRECTION #############################
            neg_clean_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] <= neg_clean_mean)
            neg_uncertain_indices = class_neg_pair_indices & class_neg_indices & (neg_clean_mean < neg_selection_metric[:, class_idx]) & (neg_selection_metric[:, class_idx] <= neg_noisy_mean)
            neg_noisy_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] > neg_noisy_mean)

            clean_mask[neg_clean_indices, class_idx] = 1  # Mark negatives as clean

            
            # For noisy label, use prediction
            labels[neg_noisy_indices, class_idx] = pred_labels[neg_noisy_indices, class_idx]

            # Soft correction for uncertain samples
            labels[neg_uncertain_indices, class_idx] = (
                soft_correction_weight * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - soft_correction_weight) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)

        
    labels = torch.tensor(labels, device = pred_labels.device)
        
    # Compute correction loss
    corr_loss = criterion_now_each(pred_labels, labels)
    criterion_ranking = MultiLabelRankingLoss(margin=1.0, reduction = 'none')
    corr_rank_loss = criterion_ranking(pred_labels, labels)
    
    final_loss = corr_loss 
    
    # Analyze correction result
    one_hot_labels = (labels > 0.5).int()
    clean_acc, corr_acc = compute_clean_noisy_accuracy(clean_mask, one_hot_labels, true_labels)
    
    orig_clean_acc, orig_noisy_acc = compute_clean_noisy_accuracy(clean_mask, original_labels, true_labels)
    print(f"Epoch {epoch}: Clean Accuracy = {clean_acc:.4f}")
    print(f"Epoch {epoch}: Noisy Accuracy = {corr_acc:.4f}")
    print(f"Epoch {epoch}: Noisy Accuracy Before Correction = {orig_noisy_acc:.4f}")
    
    
    return final_loss, corr_rank_loss, clean_mask, negative_pair_mask, labels





#### FIT GMM #####


def fit_global_gmm_and_correct_labels(args, labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric):
    """Fits one global GMM for positive and negative labels across all classes, then performs label correction."""
    batch_size, num_classes = labels.shape
    soft_correction_weight = 0.5
    
    # Flatten selection metrics across all classes
    pos_flattened = pos_selection_metric[labels > 0.5].detach().cpu().numpy()
    neg_flattened = neg_selection_metric[labels <= 0.5].detach().cpu().numpy()
    
    # Fit a single GMM for positive and negative samples
    pos_gmm = fit_gaussian_model(pos_flattened) if len(pos_flattened) > 1 else None
    neg_gmm = fit_gaussian_model(neg_flattened) if len(neg_flattened) > 1 else None

    # Extract mean thresholds from GMMs
    pos_clean_mean, pos_noisy_mean = np.sort(pos_gmm.means_.flatten()) if pos_gmm else (None, None)
    neg_clean_mean, neg_noisy_mean = np.sort(neg_gmm.means_.flatten()) if neg_gmm else (None, None)

    for class_idx in range(num_classes):
        class_pos_indices = labels[:, class_idx] > 0.5
        class_neg_indices = labels[:, class_idx] <= 0.5
        class_neg_pair_indices = negative_pair_mask[:, class_idx] == 1
        
        # Apply label correction based on global GMM thresholds
        if pos_gmm:
            pos_clean_indices = class_pos_indices & (pos_selection_metric[:, class_idx] <= pos_clean_mean)
            pos_uncertain_indices = class_pos_indices & (pos_clean_mean < pos_selection_metric[:, class_idx]) & (pos_selection_metric[:, class_idx] <= pos_noisy_mean)
            pos_noisy_indices = class_pos_indices & (pos_selection_metric[:, class_idx] > pos_noisy_mean)

            clean_mask[pos_clean_indices, class_idx] = 1  
            labels[pos_noisy_indices, class_idx] = pred_labels[pos_noisy_indices, class_idx]
            
            
            # # Probability-based soft correction for uncertain samples
            uncertainty_score = (pos_flattened - pos_clean_mean) / (pos_noisy_mean - pos_clean_mean + 1e-6)
            uncertainty_score = torch.clamp(torch.tensor(uncertainty_score, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'), 0, 1)
            
            
            labels[pos_uncertain_indices, class_idx] = (
                uncertainty_score[pos_uncertain_indices] * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - uncertainty_score[pos_uncertain_indices]) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)

        if neg_gmm:
            neg_clean_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] <= neg_clean_mean)
            neg_uncertain_indices = class_neg_pair_indices & class_neg_indices & (neg_clean_mean < neg_selection_metric[:, class_idx]) & (neg_selection_metric[:, class_idx] <= neg_noisy_mean)
            neg_noisy_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] > neg_noisy_mean)

            clean_mask[neg_clean_indices, class_idx] = 1  
            labels[neg_noisy_indices, class_idx] = pred_labels[neg_noisy_indices, class_idx]
            
            
            # # Probability-based soft correction for uncertain samples
            uncertainty_score = (neg_flattened - neg_clean_mean) / (neg_noisy_mean - neg_clean_mean + 1e-6)
            uncertainty_score = torch.clamp(torch.tensor(uncertainty_score, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'), 0, 1)  # Normalize to [0,1]
            
            
            labels[neg_uncertain_indices, class_idx] = (
                uncertainty_score[neg_uncertain_indices] * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - uncertainty_score[neg_uncertain_indices]) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)  

    return labels, clean_mask


















def fit_classwise_gmm_and_correct_labels(args, labels, pred_labels, clean_mask, negative_pair_mask, selection_metric):
    """Fits one GMM per class and performs label correction."""
    batch_size, num_classes = labels.shape
    soft_correction_weight = 0.5
    
    for class_idx in range(num_classes):
        class_pos_indices = labels[:, class_idx] > 0.5
        class_neg_indices = labels[:, class_idx] <= 0.5
        class_neg_pair_indices = negative_pair_mask[:, class_idx] == 1
        
        class_selection_metric = selection_metric[:,class_idx]
        gmm = fit_gaussian_model(class_selection_metric.detach().cpu().numpy())
        # Fit GMM for positive labels
        if len(class_selection_metric) > 1:
            
            clean_mean, noisy_mean = np.sort(gmm.means_.flatten())

            pos_clean_indices = class_pos_indices & (selection_metric[:, class_idx] <= clean_mean)
            pos_uncertain_indices = class_pos_indices & (clean_mean < selection_metric[:, class_idx]) & (selection_metric[:, class_idx] <= noisy_mean)
            pos_noisy_indices = class_pos_indices & (selection_metric[:, class_idx] > noisy_mean)

            clean_mask[pos_clean_indices, class_idx] = 1  
            
            
            
            # # Probability-based soft correction for uncertain samples
            uncertainty_score = (class_selection_metric - clean_mean) / (noisy_mean - clean_mean + 1e-6)
            uncertainty_score = torch.clamp(uncertainty_score, 0, 1)  # Normalize to [0,1]
            
            
            labels[pos_noisy_indices, class_idx] = pred_labels[pos_noisy_indices, class_idx]
            labels[pos_uncertain_indices, class_idx] = (
                uncertainty_score[pos_uncertain_indices] * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - uncertainty_score[pos_uncertain_indices]) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)

            neg_clean_indices = class_neg_pair_indices & class_neg_indices & (selection_metric[:, class_idx] <= clean_mean)
            neg_uncertain_indices = class_neg_pair_indices & class_neg_indices & (clean_mean < selection_metric[:, class_idx]) & (selection_metric[:, class_idx] <= noisy_mean)
            neg_noisy_indices = class_neg_pair_indices & class_neg_indices & (selection_metric[:, class_idx] > noisy_mean)

            clean_mask[neg_clean_indices, class_idx] = 1  
            labels[neg_noisy_indices, class_idx] = pred_labels[neg_noisy_indices, class_idx]
            labels[neg_uncertain_indices, class_idx] = (
                uncertainty_score[neg_uncertain_indices] * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - uncertainty_score[neg_uncertain_indices]) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)  
    
    return labels, clean_mask


def fit_local_gmm_and_correct_labels(args, labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric):
    batch_size, num_classes = labels.shape
    soft_correction_weight = 0.5
    
    # Iterate over each class for positive GMM fitting
    for class_idx in range(num_classes):
        
        
        
        class_pos_indices = labels[:, class_idx] > 0.5
        class_neg_indices = labels[:, class_idx] <= 0.5

        # Correct Correlated Negative Labels ONLY
        class_neg_pair_indices = negative_pair_mask[:, class_idx] == 1
        
        # Selection Metric For Positive Pair and Negative Pair Separately
        class_selection_metric = pos_selection_metric[class_pos_indices, class_idx]
        neg_class_selection_metric = neg_selection_metric[class_neg_indices, class_idx]

        if len(class_selection_metric) > 1:
            # Fit GMM for positive samples in this class
            class_selection_metric_np = class_selection_metric.detach().cpu().numpy()
            pos_gmm = fit_gaussian_model(class_selection_metric_np)
            pos_clean_mean, pos_noisy_mean = np.sort(pos_gmm.means_.flatten())
            
            
            # # Probability-based soft correction for uncertain samples
            uncertainty_score = (pos_selection_metric[:, class_idx] - pos_clean_mean) / (pos_noisy_mean - pos_clean_mean + 1e-6)
            uncertainty_score = uncertainty_score = torch.clamp(torch.tensor(uncertainty_score, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'), 0, 1) # Normalize to [0,1]
            
            
            ############################## POSITIVE CORRECTION #############################
            pos_clean_indices = class_pos_indices & (pos_selection_metric[:, class_idx] <= pos_clean_mean)
            pos_uncertain_indices = class_pos_indices & (pos_clean_mean < pos_selection_metric[:, class_idx]) & (pos_selection_metric[:, class_idx] <= pos_noisy_mean)
            pos_noisy_indices = class_pos_indices & (pos_selection_metric[:, class_idx] > pos_noisy_mean)
            
            # Update clean mask and store clean indices
            clean_mask[pos_clean_indices, class_idx] = 1  # Mark positives as clean

            
            # Probability-based soft correction for uncertain samples
            
            labels[pos_uncertain_indices, class_idx] = (
                uncertainty_score[pos_uncertain_indices] * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - uncertainty_score[pos_uncertain_indices]) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)
     

            # Hard Correction
            labels[pos_noisy_indices, class_idx] = pred_labels[pos_noisy_indices, class_idx]
            
        

        if len(neg_class_selection_metric) > 1:
            # Fit GMM for negative samples in this class
            neg_class_sm_np = neg_class_selection_metric.detach().cpu().numpy()
            neg_gmm = fit_gaussian_model(neg_class_sm_np)
            neg_clean_mean, neg_noisy_mean = np.sort(neg_gmm.means_.flatten())

            ############################## NEGATIVE CORRECTION #############################
            neg_clean_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] <= neg_clean_mean)
            neg_uncertain_indices = class_neg_pair_indices & class_neg_indices & (neg_clean_mean < neg_selection_metric[:, class_idx]) & (neg_selection_metric[:, class_idx] <= neg_noisy_mean)
            neg_noisy_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] > neg_noisy_mean)

            clean_mask[neg_clean_indices, class_idx] = 1  # Mark negatives as clean

            # labels[neg_clean_indices, class_idx] = 0
            # labels[neg_noisy_indices, class_idx] = 1  # Hard correction for noisy samples
            
            
            neg_uncertainty_score = (neg_selection_metric[:, class_idx] - neg_clean_mean) / (neg_noisy_mean - neg_clean_mean + 1e-6)
            neg_uncertainty_score = torch.clamp(neg_uncertainty_score, 0, 1)  # Normalize to [0,1]
                 
            labels[neg_uncertain_indices, class_idx] = (
                neg_uncertainty_score[neg_uncertain_indices] * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - neg_uncertainty_score[neg_uncertain_indices]) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)  
            
            labels[neg_noisy_indices, class_idx] = pred_labels[neg_noisy_indices, class_idx]

            
    return labels, clean_mask





def fit_beta_model(metric_label, n_components=2, random_state = 42):
    
    bmm = BetaMixture1D(max_iters=50).fit(metric_label.reshape(-1,1))
    
    
    # This is clean probability! ?????
    prob_clean = bmm.posterior(metric_label,bmm.means_argmin())
    prob_noisy = bmm.posterior(metric_label,bmm.means_argmax())
    

    mean_clean = bmm.means_()[bmm.means_argmin()]
    mean_noisy = bmm.means_()[bmm.means_argmax()]

    return bmm



def fit_local_bmm_and_correct_labels(args, labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric):
    batch_size, num_classes = labels.shape
    soft_correction_weight = 0.5
    
    # Iterate over each class for positive GMM fitting
    for class_idx in range(num_classes):
        class_pos_indices = labels[:, class_idx] > 0.5
        class_neg_indices = labels[:, class_idx] <= 0.5

        # Correct Correlated Negative Labels ONLY
        class_neg_pair_indices = negative_pair_mask[:, class_idx] == 1
        
        # Selection Metric For Positive Pair and Negative Pair Separately
        class_selection_metric = pos_selection_metric[class_pos_indices, class_idx]
        neg_class_selection_metric = neg_selection_metric[class_neg_indices, class_idx]

        if len(class_selection_metric) > 1:
            # Fit GMM for positive samples in this class
            class_selection_metric_np = class_selection_metric.detach().cpu().numpy()
            pos_bmm = fit_beta_model(class_selection_metric_np)
            pos_clean_mean, pos_noisy_mean = np.sort(pos_bmm.means_())
            
            ############################## POSITIVE CORRECTION #############################
            pos_clean_indices = class_pos_indices & (pos_selection_metric[:, class_idx] <= pos_clean_mean)
            pos_uncertain_indices = class_pos_indices & (pos_clean_mean < pos_selection_metric[:, class_idx]) & (pos_selection_metric[:, class_idx] <= pos_noisy_mean)
            pos_noisy_indices = class_pos_indices & (pos_selection_metric[:, class_idx] > pos_noisy_mean)
            
            # Update clean mask and store clean indices
            clean_mask[pos_clean_indices, class_idx] = 1  # Mark positives as clean


            # Hard Correction
            # labels[pos_clean_indices, class_idx] = 1
            # labels[pos_noisy_indices, class_idx] = 0  # Hard correction for noisy samples
            
            # For clean label, use original label
            # For noisy label, use prediction
            labels[pos_noisy_indices, class_idx] = pred_labels[pos_noisy_indices, class_idx]
            
            # Soft correction for uncertain samples
            labels[pos_uncertain_indices, class_idx] = (
                soft_correction_weight * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - soft_correction_weight) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)

        if len(neg_class_selection_metric) > 1:
            # Fit GMM for negative samples in this class
            neg_class_sm_np = neg_class_selection_metric.detach().cpu().numpy()
            neg_bmm = fit_beta_model(neg_class_sm_np)
            neg_clean_mean, neg_noisy_mean = np.sort(neg_bmm.means_())

            ############################## NEGATIVE CORRECTION #############################
            neg_clean_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] <= neg_clean_mean)
            neg_uncertain_indices = class_neg_pair_indices & class_neg_indices & (neg_clean_mean < neg_selection_metric[:, class_idx]) & (neg_selection_metric[:, class_idx] <= neg_noisy_mean)
            neg_noisy_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] > neg_noisy_mean)

            clean_mask[neg_clean_indices, class_idx] = 1  # Mark negatives as clean

            
            # For noisy label, use prediction
            labels[neg_noisy_indices, class_idx] = pred_labels[neg_noisy_indices, class_idx]
            

            neg_uncertainty_score = (neg_selection_metric[:, class_idx] - neg_clean_mean) / (neg_noisy_mean - neg_clean_mean + 1e-6)
            neg_uncertainty_score = torch.clamp(neg_uncertainty_score, 0, 1)  # Normalize to [0,1]
                 
            labels[neg_uncertain_indices, class_idx] = (
                neg_uncertainty_score[neg_uncertain_indices] * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - neg_uncertainty_score[neg_uncertain_indices]) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)  
        
    
    return labels, clean_mask








def fit_gaussian_model_loss_corr_latest(args, model, loss, rank, C, divers, criterion_now_each, labels, pred_labels, true_labels, epoch):
    """
    Perform correction for noisy labels using GMM fitting for positive pairs per class and a single GMM for negative pairs.

    Args:
        loss (torch.Tensor): Per-sample loss (shape: [N, D]).
        rank (torch.Tensor): Per-sample rank (shape: [N, D]).
        C (torch.Tensor): Correlation matrix (shape: [D, D]).
        criterion_now_each (function): Loss function to compute corrections.
        labels (torch.Tensor): Binary noisy labels (shape: [N, D]).
        pred_labels (torch.Tensor): Predicted probabilities (shape: [N, D]).
        true_labels (torch.Tensor): Ground truth labels (shape: [N, D]).

    Returns:
        torch.Tensor: Corrected loss.
    """
    # Normalize the selection metric
    
    
    pos_selection_metric = args.loss_coef * normalize(loss) + args.rank_coef * normalize(rank) + args.mem_coef * normalize(C)
    neg_selection_metric =  args.loss_coef * normalize(loss) + args.rank_coef * normalize(rank) + args.mem_coef * normalize(C)

        
        
    corr_labels = labels.clone()
    
        
    
    
    threshold = args.threshold_coef
    # Control EMA coefficient
    batch_size, num_classes = labels.shape

    # Compute probabilities for negative labels
    Corr = get_correlation_matrix(labels)
    negative_corr = labels @ Corr
    negative_pair_mask = (negative_corr > threshold) & (labels == 0)
    
    

    # # Initialize clean mask
    clean_mask = torch.zeros_like(labels, dtype=torch.float32)  # Adjust this weight as needed for soft correction


    if args.GlobalGMM:
        corr_labels, clean_mask = fit_global_gmm_and_correct_labels(args, corr_labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric)
    elif args.ClassGMM:   
        corr_labels, clean_mask = fit_classwise_gmm_and_correct_labels(args, corr_labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric)
    elif args.LocalGMM:
        corr_labels, clean_mask = fit_local_gmm_and_correct_labels(args, corr_labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric)  
    elif args.BMM:
        corr_labels, clean_mask = fit_local_bmm_and_correct_labels(args, corr_labels, pred_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric)  
    else:
        raise ValueError("No GMM Selection")
    

    
    
    corr_labels = torch.tensor(corr_labels, device=labels.device, dtype=torch.float64).detach()
    
        
    # Compute correction loss
    corr_loss = criterion_now_each(pred_labels, corr_labels)
    
    criterion_ranking = MultiLabelRankingLoss(margin=1.0, reduction = 'none')
    corr_rank_loss = criterion_ranking(pred_labels, corr_labels)
    
    
    
    
    # Analyze correction result
    one_hot_labels = (labels > 0.5).int()
    one_hot_corr_labels = (corr_labels > 0.5).int()
    clean_acc, corr_acc = compute_clean_noisy_accuracy(clean_mask, one_hot_corr_labels, true_labels)
    
    orig_clean_acc, orig_noisy_acc = compute_clean_noisy_accuracy(clean_mask, one_hot_labels, true_labels)
    print(f"Epoch {epoch}: Clean Accuracy = {clean_acc:.4f}")
    print(f"Epoch {epoch}: Noisy Accuracy = {corr_acc:.4f}")
    print(f"Epoch {epoch}: Noisy Accuracy Before Correction = {orig_noisy_acc:.4f}")
    
    
    return corr_loss, corr_rank_loss, clean_mask, negative_pair_mask, corr_labels







def compute_clean_noisy_accuracy(clean_mask, labels, true_labels):
    """
    Compute clean and noisy accuracy using the clean mask and corrected labels.

    Args:
        clean_mask (torch.Tensor): Mask indicating clean samples (1 = clean, 0 = noisy) [N, D].
        labels (torch.Tensor): Corrected labels after applying noise correction [N, D].
        true_labels (torch.Tensor): Ground truth labels [N, D].

    Returns:
        float: Clean accuracy (percentage of clean samples correctly classified).
        float: Noisy accuracy (percentage of noisy samples correctly classified).
    """
    # Identify clean and noisy samples
    clean_indices = clean_mask == 1
    noisy_indices = clean_mask == 0  # Noisy samples

    # Print statistics
    total_samples = clean_mask.numel()  # Total number of samples (N * D)
    total_clean_samples = clean_indices.sum().item()  # Total clean samples detected
    total_noisy_samples = noisy_indices.sum().item()  # Total noisy samples detected

    print(f"Total samples: {total_samples}")
    print(f"Total clean samples: {total_clean_samples}")
    print(f"Total noisy samples: {total_noisy_samples}")

    # Compute Clean Accuracy
    if total_clean_samples > 0:
        correct_clean_samples = (labels[clean_indices] == true_labels[clean_indices]).float().sum()
        clean_accuracy = correct_clean_samples / (total_clean_samples + 1e-6)
    else:
        clean_accuracy = torch.tensor(0.0)

    # Compute Noisy Accuracy
    if total_noisy_samples > 0:
        correct_noisy_samples = (labels[noisy_indices] == true_labels[noisy_indices]).float().sum()
        noisy_accuracy = correct_noisy_samples / (total_noisy_samples + 1e-6)
    else:
        noisy_accuracy = torch.tensor(0.0)

    return clean_accuracy.item(), noisy_accuracy.item()



def compute_correlation_matrix(labels):
    """
    Compute the correlation matrix for the given labels.

    Args:
        labels (torch.Tensor): Tensor of shape (batch_size, num_labels) with binary or continuous values.

    Returns:
        torch.Tensor: Correlation matrix of shape (num_labels, num_labels).
    """
    # Compute mean for each label
    mean_labels = labels.mean(dim=0, keepdim=True)
    
    # Compute centered labels
    centered_labels = labels - mean_labels
    
    # Compute covariance matrix
    covariance_matrix = centered_labels.T @ centered_labels / (labels.shape[0] - 1)
    
    # Compute standard deviations
    stddev = torch.sqrt(torch.diag(covariance_matrix)).unsqueeze(0)
    
    # Avoid division by zero
    stddev = torch.where(stddev == 0, torch.ones_like(stddev), stddev)
    
    # Compute correlation matrix
    correlation_matrix = covariance_matrix / (stddev.T @ stddev)
    return correlation_matrix

class MultiLabelCorrelationLoss(nn.Module):
    """
    Multi-label correlation loss that aligns predicted and true correlation matrices.

    Args:
        reduction (str): Reduction method ('mean' or 'sum').
    """
    def __init__(self, reduction='mean'):
        super(MultiLabelCorrelationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Compute the correlation loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities of shape (batch_size, num_labels).
            y_true (torch.Tensor): Ground truth labels of shape (batch_size, num_labels).

        Returns:
            torch.Tensor: Correlation loss value.
        """
        # Compute ground truth and predicted correlation matrices
        corr_true = compute_correlation_matrix(y_true)
        corr_pred = compute_correlation_matrix(y_pred)
        
        # Compute the Frobenius norm of the difference
        loss = torch.norm(corr_true - corr_pred, p='fro')
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss / y_true.size(1)  # Normalize by number of labels
        elif self.reduction == 'sum':
            pass  # Keep the sum as is

        return loss





from tsaug import TimeWarp

import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random



from tsaug import TimeWarp
import torch



def pad_to_length(tensor, length, pad_value=0.0):
    """
    Pad or truncate a tensor to a fixed length along the time dimension.

    :param tensor: Tensor of shape (batch_size, time_steps, features).
    :param length: Desired fixed length for the time dimension.
    :param pad_value: Value to use for padding. Default is 0.0.
    :return: Padded or truncated tensor of shape (batch_size, length, features).
    """
    batch_size, time_steps, features = tensor.shape
    if time_steps < length:
        pad_size = length - time_steps
        tensor = F.pad(tensor, (0, 0, 0, pad_size), value=pad_value)
    else:
        tensor = tensor[:, :length, :]
    return tensor

def augment_inst(ehr, note, max_length=513):
    """
    Augment EHR and note data, and pad or truncate to fixed length.
    
    :param ehr: A torch tensor of shape (batch_size, time_steps, features).
    :param note: A batch of note data (list of strings).
    :param max_length: Fixed length for time dimension in EHR and notes.
    :return: Augmented and padded EHR and note data.
    """
    # Convert EHR tensor to numpy for augmentation
    ehr_np = ehr.cpu().numpy()

    # Augment EHR data with TimeWarping
    augmented_ehr = []
    for idx, series in enumerate(ehr_np):
        augmented_series = ehr_augmenter.augment(series)
        augmented_ehr.append(augmented_series)

    # Convert back to torch tensor
    augmented_ehr = torch.tensor(augmented_ehr, dtype=torch.float32).to(ehr.device)


    # Augment Note data with TextAttack's WordNet Augmenter
    augmented_note = [text_augmenter.augment(sentence)[0] if sentence else "" for sentence in note]
    return augmented_ehr, augmented_note



def apply_label_smoothing(labels, alpha=0.1):
    """
    Apply label smoothing for binary labels.

    Args:
        labels (torch.Tensor): Original labels, shape (n_samples, n_classes).
        alpha (float): Smoothing factor.

    Returns:
        torch.Tensor: Smoothed labels.
    """
    smoothed_labels = (1 - alpha) * labels + alpha / 2  # Binary smoothing
    return smoothed_labels




def bpr_loss_all_pairs(y_pred, y_true):
    """
    Computes the BPR loss using all possible positive-negative pairs without balancing.

    Args:
        y_pred (torch.Tensor): Predicted logits (N, C).
        y_true (torch.Tensor): Ground truth binary labels (N, C).

    Returns:
        torch.Tensor: Computed BPR loss.
    """
    y_pred_prob = torch.sigmoid(y_pred)

    pos_mask = (y_true == 1).unsqueeze(2)
    neg_mask = (y_true == 0).unsqueeze(1)

    pos_preds = y_pred_prob.unsqueeze(2)  # Shape (N, C, 1)
    neg_preds = y_pred_prob.unsqueeze(1)  # Shape (N, 1, C)

    pairwise_diff = pos_preds - neg_preds  # Compute all pairs

    loss = -torch.log(torch.sigmoid(pairwise_diff) + 1e-8)  # Add small epsilon to avoid log(0)

    return loss.mean()



def compute_cooccurrence_matrix(labels):
    """
    Compute the co-occurrence matrix for multi-label data.
    
    Args:
        labels (torch.Tensor): Binary label matrix of shape (batch_size, n_bins),
                               where each entry is 0 or 1.

    Returns:
        co_matrix (torch.Tensor): Co-occurrence matrix of shape (n_bins, n_bins).
    """
    batch_size, n_bins = labels.shape
    # Convert to float for matrix multiplication
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.float32, device='cuda')  # Assuming GPU usage
    else:
        labels = labels.float()
    
    # Co-occurrence matrix: (n_bins, n_bins)
    co_matrix = torch.matmul(labels.T, labels) / batch_size
    
    # Normalize diagonals to 1
    co_matrix.fill_diagonal_(1.0)
    
    return co_matrix




# def visualization(epoch, loss_all, mems_all, ranks_all, pred_all, divers_all, true_all, noisy_all, stats_over_epochs, save_dir):
#     """
#     Visualizes loss, mems, predictions, ranks, and diversity for clean and noisy labels for a given epoch.
#     Records statistics across epochs and saves line plots for each metric.

#     Parameters:
#     - epoch (int): The epoch number for visualization.
#     - loss_all (np.ndarray): Loss of all samples (samples x metrics x epochs).
#     - mems_all (np.ndarray): Memory of all samples (samples x metrics x epochs).
#     - pred_all (np.ndarray): Predictions of all samples (samples x metrics x epochs).
#     - ranks_all (np.ndarray): Ranks of all samples (samples x metrics x epochs).
#     - divers_all (np.ndarray): Diversity of all samples (samples x metrics x epochs).
#     - true_all (np.ndarray): True (clean) labels for all samples.
#     - noisy_all (np.ndarray): Noisy labels for all samples.
#     - stats_over_epochs (dict): Dictionary to record statistics across epochs.
#     - save_dir (str): Directory to save visualizations.
#     """

#     # Extract data for the specified epoch
#     loss = normalize(loss_all[:, :, epoch - 1])
#     mems = normalize(mems_all[:, :, epoch - 1])
#     pred = normalize(pred_all[:, :, epoch - 1])
#     ranks = normalize(ranks_all[:, :, epoch - 1])
#     divs = normalize(divers_all[:, :, epoch - 1])
#     true = true_all[:, :, epoch - 1]
#     noisy = noisy_all[:, :, epoch - 1]
    
    
#     Z = 0.5 * mems + 0.5 * ranks

#     # Create masks for clean and noisy labels
#     clean_mask = true == noisy  # Clean samples where true label matches the noisy label
#     noisy_mask = true != noisy  # Noisy samples where true label differs from the noisy label

#     # Positive and negative pair masks
#     positive_pair_mask = noisy == 1  # Observed noisy label == 1
#     negative_pair_mask = noisy == 0  # Observed noisy label == 0

#     # Subdivide positive/negative pairs into clean and noisy
#     positive_clean_mask = positive_pair_mask & clean_mask  # Positive pair and clean
#     positive_noisy_mask = positive_pair_mask & noisy_mask  # Positive pair and noisy
#     negative_clean_mask = negative_pair_mask & clean_mask  # Negative pair and clean
#     negative_noisy_mask = negative_pair_mask & noisy_mask  # Negative pair and noisy

#     # Compute statistics for all categories
#     metrics = {"Loss": loss, "Mems": mems, "Predictions": pred, "Ranks": ranks, "Diversity": divs}
#     stats = {}
#     for name, data in metrics.items():
#         stats[name] = {
#             "Positive Pair Clean Mean": np.mean(data[positive_clean_mask]),
#             "Positive Pair Clean SD": np.std(data[positive_clean_mask]),
#             "Positive Pair Noisy Mean": np.mean(data[positive_noisy_mask]),
#             "Positive Pair Noisy SD": np.std(data[positive_noisy_mask]),
#             "Negative Pair Clean Mean": np.mean(data[negative_clean_mask]),
#             "Negative Pair Clean SD": np.std(data[negative_clean_mask]),
#             "Negative Pair Noisy Mean": np.mean(data[negative_noisy_mask]),
#             "Negative Pair Noisy SD": np.std(data[negative_noisy_mask]),
#         }
        
        
        

#     # Record statistics for the current epoch
#     for metric, stat in stats.items():
#         stats_over_epochs[metric].append(stat)

#     # Generate and save plots for each metric
#     os.makedirs(save_dir, exist_ok=True)
#     for metric, data in metrics.items():
#         plt.figure(figsize=(10, 6))

#         plt.hist(data[positive_clean_mask].flatten(), bins=20, alpha=0.6, label='Positive Pair Clean', color='blue')
#         plt.hist(data[positive_noisy_mask].flatten(), bins=20, alpha=0.6, label='Positive Pair Noisy', color='orange')
#         plt.hist(data[negative_clean_mask].flatten(), bins=20, alpha=0.6, label='Negative Pair Clean', color='green')
#         plt.hist(data[negative_noisy_mask].flatten(), bins=20, alpha=0.6, label='Negative Pair Noisy', color='red')

#         # Make title bolder and larger
#         plt.title(f'{metric} Distribution (Epoch {epoch})', fontsize=16, fontweight='bold')
#         plt.xlabel(f'{metric} Value', fontsize=14)
#         plt.ylabel('Frequency', fontsize=14)

#         # Make legend larger
#         plt.legend(fontsize=12, loc='upper right')

#         plt.grid()

#         # Save plot
#         save_path = os.path.join(save_dir, f'{metric}_epoch_{epoch}.png')
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save
#         plt.close()

#     return stats_over_epochs
    


# def plot_statistics(stats_over_epochs, save_dir):
#     """
#     Plot statistics across all epochs for each metric.

#     Parameters:
#     - stats_over_epochs (dict): Recorded statistics across epochs.
#     - save_dir (str): Directory to save plots.
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     for metric, stats_list in stats_over_epochs.items():
#         epochs = range(1, len(stats_list) + 1)
#         positive_clean = [stat["Positive Pair Clean Mean"] for stat in stats_list]
#         positive_noisy = [stat["Positive Pair Noisy Mean"] for stat in stats_list]
#         negative_clean = [stat["Negative Pair Clean Mean"] for stat in stats_list]
#         negative_noisy = [stat["Negative Pair Noisy Mean"] for stat in stats_list]

#         plt.figure(figsize=(10, 6))
#         plt.plot(epochs, positive_clean, label='Positive Pair Clean', marker='o', linestyle='-', linewidth=2)
#         plt.plot(epochs, positive_noisy, label='Positive Pair Noisy', marker='o', linestyle='--', linewidth=2)
#         plt.plot(epochs, negative_clean, label='Negative Pair Clean', marker='s', linestyle='-', linewidth=2)
#         plt.plot(epochs, negative_noisy, label='Negative Pair Noisy', marker='s', linestyle='--', linewidth=2)

#         # Make title larger and bold
#         plt.title(f'{metric} Across Epochs', fontsize=18, fontweight='bold', fontname='Arial')

#         plt.xlabel('Epoch', fontsize=14, fontweight='bold')
#         plt.ylabel(f'{metric} Mean Value', fontsize=14, fontweight='bold')

#         # Make legend larger
#         plt.legend(fontsize=13, loc='upper right', frameon=True)

#         plt.grid()

#         # Save plot with high resolution
#         save_path = os.path.join(save_dir, f'{metric}_across_epochs.png')
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"Saved {metric} plot to {save_path}")
        


import os
import matplotlib.pyplot as plt

def plot_statistics(stats_over_epochs, save_dir):
    """
    Plot clean vs. noisy statistics across all epochs for each metric.

    Parameters:
    - stats_over_epochs (dict): Recorded statistics across epochs.
    - save_dir (str): Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for metric, stats_list in stats_over_epochs.items():
        epochs = range(1, len(stats_list) + 1)

        clean_means = [stat["Clean Mean"] for stat in stats_list]
        noisy_means = [stat["Noisy Mean"] for stat in stats_list]
        clean_stds = [stat["Clean SD"] for stat in stats_list]
        noisy_stds = [stat["Noisy SD"] for stat in stats_list]

        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, clean_means, label='Clean', marker='o', linestyle='-', linewidth=2, color='blue')
        plt.fill_between(epochs,
                         [m - s for m, s in zip(clean_means, clean_stds)],
                         [m + s for m, s in zip(clean_means, clean_stds)],
                         color='blue', alpha=0.2)

        plt.plot(epochs, noisy_means, label='Noisy', marker='s', linestyle='--', linewidth=2, color='red')
        plt.fill_between(epochs,
                         [m - s for m, s in zip(noisy_means, noisy_stds)],
                         [m + s for m, s in zip(noisy_means, noisy_stds)],
                         color='red', alpha=0.2)

        plt.title(f'{metric} Across Epochs', fontsize=18, fontweight='bold', fontname='Arial')
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel(f'{metric} Mean Value', fontsize=14, fontweight='bold')

        plt.legend(fontsize=13, loc='upper right', frameon=True)
        plt.grid(True)

        save_path = os.path.join(save_dir, f'{metric}_across_epochs.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric} plot to {save_path}")







