from sklearn.mixture import GaussianMixture
import numpy as np
import torch
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


def partition_metric_into_sets(metric_label, gmm_params):
    """
    Partition a metric into clean, uncertain, and noisy sets based on GMM means.

    Args:
        metric_label (array-like): Metric values for a specific label group.
        gmm_params (dict): Fitted GMM parameters containing means, weights, and covariances.

    Returns:
        dict: Partitioned sets {'clean': [...], 'uncertain': [...], 'noisy': [...]}.
    """
    # Extract GMM means and sort them
    clean_mean, noisy_mean = np.sort(gmm_params["means"])

    # Partition the data
    clean_set = metric_label[metric_label <= clean_mean]
    uncertain_set = metric_label[(metric_label > clean_mean) & (metric_label < noisy_mean)]
    noisy_set = metric_label[metric_label >= noisy_mean]

    return {"clean": clean_set, "uncertain": uncertain_set, "noisy": noisy_set}


def fit_gaussian_model(metric_label, n_components=2, random_state = 42):
    # Fit Gaussian Mixture Models
    
    gmm = GaussianMixture(n_components=n_components, random_state=42).fit(
        metric_label.reshape(-1, 1))

    return gmm





def get_correlation_matrix(labels):
    # Get Correlation Matrix
    C = labels.T @ labels
    C.fill_diagonal_(0)
    row_sums = C.sum(dim=1, keepdim=True)  # Sum along rows
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    C = C / row_sums
    return C


def compute_pos_label_prob(labels, metrics, pos_gmm):
    """
    Compute probabilities for each label using positive and negative GMMs (PyTorch version).
    
    Args:
        labels (torch.Tensor): Binary label matrix (128 x 25).
        metrics (torch.Tensor): Selection metric matrix (128 x 25).
        pos_gmm (GaussianMixture): GMM fitted on positive label metrics.
        neg_gmm (GaussianMixture): GMM fitted on negative label metrics.
    
    Returns:
        torch.Tensor: Probability matrix (128 x 25).
    """
    # Flatten the labels and metrics
    labels_flat = labels.view(-1)  # Shape: (128 * 25,)
    metrics_flat = metrics.view(-1)  # Shape: (128 * 25,)
    
    metrics_flat_numpy = metrics_flat.detach().cpu().numpy()


    # Compute probabilities for each label
    probabilities_flat = torch.zeros_like(metrics_flat, dtype=torch.float32)  # Initialize as zeros

    # Positive labels: Use pos_gmm
    pos_indices = (labels_flat == 1).nonzero(as_tuple=True)[0]  # Indices of positive labels
    if len(pos_indices) > 0:
        pos_probs = pos_gmm.predict_proba(metrics_flat_numpy[pos_indices.cpu().numpy()].reshape(-1,1))[:, pos_gmm.means_.argmin()]
        probabilities_flat[pos_indices] = torch.tensor(pos_probs, dtype=torch.float32, device=labels.device)

    probabilities = probabilities_flat.view(labels.shape)
    return probabilities





def compute_label_probabilities(labels, metrics, pos_gmm, neg_gmm):
    """
    Compute probabilities for each label using positive and negative GMMs (PyTorch version).
    
    Args:
        labels (torch.Tensor): Binary label matrix (128 x 25).
        metrics (torch.Tensor): Selection metric matrix (128 x 25).
        pos_gmm (GaussianMixture): GMM fitted on positive label metrics.
        neg_gmm (GaussianMixture): GMM fitted on negative label metrics.
    
    Returns:
        torch.Tensor: Probability matrix (128 x 25).
    """
    # Flatten the labels and metrics
    labels_flat = labels.view(-1)  # Shape: (128 * 25,)
    metrics_flat = metrics.view(-1)  # Shape: (128 * 25,)
    
    metrics_flat_numpy = metrics_flat.detach().cpu().numpy()


    # Compute probabilities for each label
    probabilities_flat = torch.zeros_like(metrics_flat, dtype=torch.float32)  # Initialize as zeros

    # Positive labels: Use pos_gmm
    pos_indices = (labels_flat == 1).nonzero(as_tuple=True)[0]  # Indices of positive labels
    if len(pos_indices) > 0:
        pos_probs = pos_gmm.predict_proba(metrics_flat_numpy[pos_indices.cpu().numpy()].reshape(-1,1))[:, pos_gmm.means_.argmin()]
        probabilities_flat[pos_indices] = torch.tensor(pos_probs, dtype=torch.float32, device=labels.device)

    # Negative labels: Use neg_gmm
    neg_indices = (labels_flat == 0).nonzero(as_tuple=True)[0]  # Indices of negative labels
    if len(neg_indices) > 0:
        neg_probs = neg_gmm.predict_proba(metrics_flat_numpy[neg_indices.cpu().numpy()].reshape(-1,1))[:, neg_gmm.means_.argmin()]
        probabilities_flat[neg_indices] = torch.tensor(neg_probs, dtype=torch.float32, device=labels.device)

    # Reshape back to original dimensions
    probabilities = probabilities_flat.view(labels.shape)
    return probabilities