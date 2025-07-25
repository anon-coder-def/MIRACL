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



####### Data Preprocessing



def my_collate(batch):
    # Time series data  (When missing, use an all-zero vector with shape of [1,76])
    ehr = [item[0][-512:] if np.array_equal(item[0], None) is False else np.zeros((1,76)) for item in batch]
    ehr, ehr_length = pad_zeros(ehr)
    mask_ehr = np.array([1 if np.array_equal(item[0], None) is False else 0 for item in batch])     # Marks whether EHR is included
    ehr_length = [0 if mask_ehr[i] == 0 else ehr_length[i] for i in range(len(ehr_length))]  # Remove fictitious time series
    
    

    # Note text data    (An empty string has been used to indicate modality missing)
    note = [item[1] for item in batch]
    mask_note = np.array([1 if item[1] != '' else 0 for item in batch])

    # Label
    label = np.array([item[2] for item in batch]).reshape(len(batch),-1)
    
    
    # Label
    noisy_label = np.array([item[3] for item in batch]).reshape(len(batch),-1)

    # Task
    replace_dict = {'in-hospital-mortality':0, 'decompensation':1, 'phenotyping':2, 'length-of-stay':3, 'readmission':4, 'diagnosis':5, 'drg':6}
    task_index = np.array([replace_dict[item[6]] if item[6] in replace_dict else -1 for item in batch])
    
    
    # Row Index (Instance IDs)
    x_ids = [item[7] for item in batch]
    
    patient_id = np.array([item[8] for item in batch])
    
    
    return [ehr, ehr_length, mask_ehr, note, mask_note, label, noisy_label, task_index, x_ids, patient_id]



# Pad the time series to the same length
def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in ret]
    return np.array(ret), seq_length


def read_timeseries(args):
    if args.dataset == 'mimic3':
        path = f'{args.ehr_path}/3_episode1_timeseries.csv'
    elif args.dataset == 'mimic4':
        path = f'{args.ehr_path}/10002430_episode1_timeseries.csv'
    else:
        raise Exception("no available dataset")

    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)


####### Normalisation


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


            
            labels[pos_uncertain_indices, class_idx] = (
                uncertainty_score[pos_uncertain_indices] * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - uncertainty_score[pos_uncertain_indices]) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)
     

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

            
            neg_uncertainty_score = (neg_selection_metric[:, class_idx] - neg_clean_mean) / (neg_noisy_mean - neg_clean_mean + 1e-6)
            neg_uncertainty_score = torch.clamp(neg_uncertainty_score, 0, 1)  # Normalize to [0,1]
                 
            labels[neg_uncertain_indices, class_idx] = (
                neg_uncertainty_score[neg_uncertain_indices] * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - neg_uncertainty_score[neg_uncertain_indices]) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)  
            
            labels[neg_noisy_indices, class_idx] = pred_labels[neg_noisy_indices, class_idx]
            
            
    return labels, clean_mask




# FIT AND CORRECT

def fit_gaussian_model_loss_corr_latest(args, model, loss, rank, C, score, ehr, note, criterion_now_each, labels, pred_labels, true_labels, epoch):
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
    
    
    pos_selection_metric = args.loss_coef * normalize(loss) + args.rank_coef * normalize(rank) + args.mem_coef * normalize(C) * args.score_coef * score
    neg_selection_metric =  args.loss_coef * normalize(loss) + args.rank_coef * normalize(rank) + args.mem_coef * normalize(C) * args.score_coef * score
    
        
        
    corr_labels = labels.clone()
    pred_star_labels = pred_labels.clone()
    
        
    pos_selection_metric = normalize(pos_selection_metric)
    neg_selection_metric = normalize(neg_selection_metric)
    
    
    threshold = args.threshold_coef
    # Control EMA coefficient
    batch_size, num_classes = labels.shape

    # Compute probabilities for negative labels
    Corr = get_correlation_matrix(labels)
    negative_corr = labels @ Corr
    negative_pair_mask = (negative_corr > threshold) & (labels == 0)
    
    
    # Add EHR and Note prediction logic
    
    pred_star_labels = 0.6 * pred_labels + 0.2 * ehr + 0.2 * note
    

    # # Initialize clean mask
    clean_mask = torch.zeros_like(labels, dtype=torch.float32)  # Adjust this weight as needed for soft correction


    if args.GlobalGMM:
        corr_labels, clean_mask = fit_global_gmm_and_correct_labels(args, corr_labels, pred_star_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric)
    elif args.ClassGMM:   
        corr_labels, clean_mask = fit_classwise_gmm_and_correct_labels(args, corr_labels, pred_star_labels, clean_mask, negative_pair_mask, pos_selection_metric)
    elif args.LocalGMM:
        corr_labels, clean_mask = fit_local_gmm_and_correct_labels(args, corr_labels, pred_star_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric)  
    elif args.BMM:
        corr_labels, clean_mask = fit_local_bmm_and_correct_labels(args, corr_labels, pred_star_labels, clean_mask, negative_pair_mask, pos_selection_metric, neg_selection_metric)  
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




    # 四类索引
    clean_pos = (true_labels == 1) & (labels == 1)
    noisy_pos = (true_labels == 0) & (labels == 1)
    clean_neg = (true_labels == 0) & (labels == 0)
    noisy_neg = (true_labels == 1) & (labels == 0)

    def compute_accuracy(mask, name=''):
        total = mask.sum()
        if total > 0:
            correct = (labels[mask] == true_labels[mask]).float().sum()
            acc = correct / (total + 1e-6)
        else:
            acc = torch.tensor(0.0)
        return acc

    # 分别计算四类准确率
    acc_clean_pos = compute_accuracy(clean_pos, 'Clean+')
    acc_noisy_pos = compute_accuracy(noisy_pos, 'Noisy+')
    acc_clean_neg = compute_accuracy(clean_neg, 'Clean-')
    acc_noisy_neg = compute_accuracy(noisy_neg, 'Noisy-')

    # 可选：整理为 dict 输出
    accuracy_by_type = {
        'Clean+': acc_clean_pos.item(),
        'Noisy+': acc_noisy_pos.item(),
        'Clean-': acc_clean_neg.item(),
        'Noisy-': acc_noisy_neg.item()
    }




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

