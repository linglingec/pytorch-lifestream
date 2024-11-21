import torch
import torch.nn as nn
import torch.nn.functional as F

def np_loss(np_numerical_outputs, np_categorical_outputs, target_numerical, target_categorical):
    """
    Computes the Next Prediction (NP) task loss.

    Args:
        np_numerical_outputs (Tensor): Predicted numerical values for NP task, shape (batch_size, seq_len - 1, num_numerical_features).
        np_categorical_outputs (list of Tensor): Predicted categorical probabilities for NP task, 
                                                 each of shape (batch_size, seq_len - 1, num_categories).
        target_numerical (Tensor): Ground truth numerical values, shape (batch_size, seq_len - 1, num_numerical_features).
        target_categorical (list of Tensor): Ground truth categorical labels for NP task, 
                                             each of shape (batch_size, seq_len - 1).
    
    Returns:
        Tensor: The NP task loss.
    """
    # MSE loss for numerical features
    numerical_loss = F.mse_loss(np_numerical_outputs, target_numerical, reduction="mean")

    # Cross-entropy loss for categorical features
    categorical_loss = sum(
        F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),  # Flatten predictions for each feature
            target.reshape(-1),  # Flatten ground truth labels
            reduction="mean"
        )
        for pred, target in zip(np_categorical_outputs, target_categorical)
    ) / len(np_categorical_outputs)

    return numerical_loss + categorical_loss

def pr_loss(pr_numerical_outputs, pr_categorical_outputs, target_numerical, target_categorical, time_gaps, lambda_param=1.0):
    """
    Computes the Past Reconstruction (PR) task loss with exponential decay for time gaps.

    Args:
        pr_numerical_outputs (Tensor): Predicted numerical values for PR task, shape (batch_size, seq_len - 1, max_past_events, num_numerical_features).
        pr_categorical_outputs (list of Tensor): Predicted categorical probabilities for PR task, 
                                                 each of shape (batch_size, seq_len - 1, max_past_events, num_categories).
        target_numerical (Tensor): Ground truth numerical values for past events, 
                                   shape (batch_size, seq_len - 1, max_past_events, num_numerical_features).
        target_categorical (list of Tensor): Ground truth categorical labels for past events, 
                                             each of shape (batch_size, seq_len - 1, max_past_events).
        time_gaps (Tensor): Time differences between current and past events, 
                            shape (batch_size, seq_len - 1, max_past_events), with -1 indicating padding.
        lambda_param (float): Decay parameter for time gap weighting.
    
    Returns:
        Tensor: The PR task loss.
    """
    # Mask for valid time gaps
    valid_time_mask = (time_gaps != -1)

    # Compute exponential weights
    time_weights = torch.exp(-time_gaps / lambda_param) * valid_time_mask.float()

    # Numerical loss: Compute MSE per event and apply weights after reduction
    mse_loss_per_event = F.mse_loss(
        pr_numerical_outputs, target_numerical, reduction="none"
    )  # Shape: (batch_size, seq_len, max_past_events, num_numerical_features)

    mse_loss_weighted = (mse_loss_per_event.sum(dim=-1) * time_weights).sum()  # Weighted sum over past events

    # Categorical loss: Compute cross-entropy per event and apply weights after reduction
    categorical_loss_weighted = 0.0
    for pred, target in zip(pr_categorical_outputs, target_categorical):
        # Flatten dimensions for processing
        pred_flat = pred.reshape(-1, pred.size(-1))  # Shape: (batch_size * seq_len * max_past_events, num_categories)
        target_flat = target.reshape(-1)  # Shape: (batch_size * seq_len * max_past_events)
        valid_mask_flat = valid_time_mask.reshape(-1)  # Shape: (batch_size * seq_len * max_past_events)
        weights_flat = time_weights.reshape(-1)  # Shape: (batch_size * seq_len * max_past_events)

        # Compute per-event cross-entropy loss
        ce_loss_per_event = F.cross_entropy(pred_flat, target_flat, reduction="none")
        ce_loss_weighted = (ce_loss_per_event * weights_flat * valid_mask_flat).sum()

        categorical_loss_weighted += ce_loss_weighted

    return mse_loss_weighted + categorical_loss_weighted

def nppr_loss(np_numerical_outputs, np_categorical_outputs, pr_numerical_outputs, pr_categorical_outputs, target_np_numerical, 
                target_np_categorical, target_pr_numerical, target_pr_categorical, time_gaps, lambda_param=1.0, alpha=1.0):
    """
    Computes the combined NPPR loss, combining NP loss and PR loss.

    Args:
        np_numerical_outputs (Tensor): Predicted numerical values for NP task, shape (batch_size, seq_len, num_numerical_features).
        np_categorical_outputs (list of Tensor): Predicted categorical probabilities for NP task, 
                                                 each of shape (batch_size, seq_len, num_categories).
        pr_numerical_outputs (Tensor): Predicted numerical values for PR task, shape (batch_size, seq_len, max_past_events, num_numerical_features).
        pr_categorical_outputs (list of Tensor): Predicted categorical probabilities for PR task, 
                                                 each of shape (batch_size, seq_len, max_past_events, num_categories).
        target_np_numerical (Tensor): Ground truth numerical values for NP task, shape (batch_size, seq_len, num_numerical_features).
        target_np_categorical (list of Tensor): Ground truth categorical labels for NP task, 
                                                each of shape (batch_size, seq_len).
        target_pr_numerical (Tensor): Ground truth numerical values for PR task, shape (batch_size, seq_len, max_past_events, num_numerical_features).
        target_pr_categorical (list of Tensor): Ground truth categorical labels for PR task, 
                                                each of shape (batch_size, seq_len, max_past_events).
        time_gaps (Tensor): Time differences between current and past events, 
                            shape (batch_size, seq_len, max_past_events), with -1 indicating padding.
        lambda_param (float): Decay parameter for time gap weighting in PR loss.
        alpha (float): Weight for combining NP and PR losses. Default is 1.0.

    Returns:
        Tensor: The combined NPPR loss.
    """
    # Compute NP loss
    np_loss_value = np_loss(np_numerical_outputs, np_categorical_outputs, target_np_numerical, target_np_categorical)

    # Compute PR loss
    pr_loss_value = pr_loss(pr_numerical_outputs, pr_categorical_outputs, target_pr_numerical, target_pr_categorical, time_gaps, lambda_param)

    # Combine NP and PR losses
    total_loss = (1 - alpha) * np_loss_value + alpha * pr_loss_value

    return total_loss
