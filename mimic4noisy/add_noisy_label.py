import os
import numpy as np
import pandas as pd
import argparse
import torch
from scipy import stats
from math import inf
import torch.nn.functional as F
# python mimic4noisy/add_noisy_label.py --tasks in-hospital-mortality --noise_ratio 0.5 --noise_type Symm


# Function to induce label noise in a single-label setting with symmetric or asymmetric noise
def flip_label(t, target, noise_ratio, noise_type, args):
    
    np.random.seed(123)
    assert 0 <= noise_ratio < 1

    target = np.array(target).astype(int)
    
    label = target.copy()
    n_class = args.nbins[t]
    if abs(noise_ratio - 0) < 1e-5:
        return label, np.array([int(x != y) for (x, y) in zip(target, label)])

    for i in range(label.shape[0]):
        if noise_type == 'Symm':
            # Symmetric noise: equal probability of flipping to any other class
            p1 = np.ones(n_class) * noise_ratio / (n_class - 1)
            p1[label[i]] = 1 - noise_ratio
            label[i] = np.random.choice(n_class, p=p1)
        elif noise_type == 'Asym':
            # Asymmetric noise: structured noise where each label flips to a specific class
            label[i] = np.random.choice([label[i], (label[i] + 1) % n_class], p=[1 - noise_ratio, noise_ratio])
              
    mask = np.array([int(x != y) for (x, y) in zip(target, label)])
    return label, mask



# Function to induce label noise in a multi-label setting with symmetric or asymmetric transition
def flip_multilabel(t, target, noise_ratio, noise_type, args):
    """
    Induce label noise in a multi-label setting with symmetric, asymmetric, balanced, or class-conditional noise.

    Args:
        t (int): Current task identifier.
        target (np.ndarray): Binary label matrix of shape (num_samples, num_classes).
        noise_ratio (float or list): Noise intensity.
        noise_type (str): Type of noise ('Symm', 'Asym', 'Balanced', 'ClassConditional').
        args: Additional arguments, including the number of bins (nbins).

    Returns:
        noisy_labels (np.ndarray): Noisy labels after flipping.
        mask (np.ndarray): Mask indicating which labels were flipped.
    """
    np.random.seed(123)
    
    target = np.array(target, dtype=int)
    
    # if target contain invalid value, replace it with 0
    target[target <= -1] = 0
    
    n_class = args.nbins[t]  # Number of labels per instance
    noisy_labels = target.copy()
    mask = np.zeros_like(target)  # Mask to track which labels were flipped
    
    # Calculate class counts and total samples
    class_counts = np.sum(target, axis=0)  # N_i for each class
    total_samples = len(target)  # Total number of samples

    inst_counts = np.sum(target, axis=1)
    L_avg = np.mean(inst_counts)
    bal_ratio = noise_ratio[0]*L_avg/(n_class-L_avg + 1e-8)
    total_samples = len(target)
    print(L_avg)
    print("Bal_Ratio")
    print(bal_ratio)
    
    # Loop over each class and instance
    for label_col in range(n_class):
        for i in range(len(target)):
            original_label = target[i, label_col]
            
            # Define flipping logic
            if noise_type == 'Symm':
                # Symmetric noise: flip with equal probability
                flip_prob = np.array([1 - noise_ratio, noise_ratio])
                new_label = np.random.choice([original_label, 1 - original_label], p=flip_prob)
            
            elif noise_type == 'Asym':
                # Asymmetric noise
                if original_label == 1:
                    new_label = np.random.choice([1, 0], p=[1 - noise_ratio[0], noise_ratio[0]])
                else:
                    new_label = np.random.choice([0, 1], p=[1 - noise_ratio[1], noise_ratio[1]])
            
            elif noise_type == 'Balanced':
                # Balanced noise
                if original_label == 1:
                    new_label = np.random.choice([1, 0], p=[1 - noise_ratio[0], noise_ratio[0]])
                else:
                    new_label = np.random.choice([0, 1], p=[1 - bal_ratio, bal_ratio])
            
            elif noise_type == 'CCN':
                # Class-conditional noise
                if target[i, label_col] == 1:  # Only flip if the original label is 1
                    # Calculate conditional probabilities for flipping to other classes
                    Ni = class_counts[label_col]  # Number of positives for the current class
                    Nj = np.delete(class_counts, label_col)  # Positive counts for other classes
                    N_minus_Ni = total_samples - Ni  # Total minus positives for the current class
                    
                    # Compute probabilities for flipping to each other class
                    conditional_probs = noise_ratio[0] * Nj / (N_minus_Ni + 1e-8)
                    conditional_probs /= np.sum(conditional_probs)  # Normalize probabilities
                    
                    # Randomly select a new class based on probabilities
                    new_class_idx = np.random.choice(range(n_class - 1), p=conditional_probs)
                    new_class = np.delete(np.arange(n_class), label_col)[new_class_idx]

                    # Flip the label: Set the current class to 0 and the new class to 1
                    noisy_labels[i, label_col] = 0
                    noisy_labels[i, new_class] = 1
                    
                    # Update the mask to indicate a flip occurred
                    mask[i, label_col] = 1
                    mask[i, new_class] = 1
                continue  # Skip further updates for this label

            # Update noisy_labels and mask for other noise types
            noisy_labels[i, label_col] = new_label
            if original_label != new_label:
                mask[i, label_col] = 1  # Mark flip in mask

    if mask.sum() == 0:
        print("Warning: No flips occurred. Check noise ratio and flipping logic.")
    
    return noisy_labels, mask
