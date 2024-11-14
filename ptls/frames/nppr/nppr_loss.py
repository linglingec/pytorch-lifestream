import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss functions for NP, PR, and combined NPPR losses
def np_loss(np_numerical_outputs, np_categorical_outputs, target_numerical, target_categorical):
    """
    Computes the Next Prediction (NP) task loss.

    Args:
        np_numerical_outputs (Tensor): Predicted numerical values for NP task.
        np_categorical_outputs (list of Tensor): Predicted categorical probabilities for NP task.
        target_numerical (Tensor): Ground truth numerical values.
        target_categorical (list of Tensor): Ground truth categorical labels.
    
    Returns:
        Tensor: The NP task loss.
    """
    # MSE loss for numerical features
    numerical_loss = F.mse_loss(np_numerical_outputs, target_numerical, reduction='mean')
    
    # Cross-entropy loss for categorical features
    categorical_loss = sum(F.cross_entropy(pred, target, reduction='mean') 
                           for pred, target in zip(np_categorical_outputs, target_categorical)) / len(np_categorical_outputs)
    
    return numerical_loss + categorical_loss


def pr_loss(pr_numerical_outputs, pr_categorical_outputs, target_numerical, target_categorical, time_gaps, lambda_param=1.0, max_past_events=5):
    """
    Computes the Past Reconstruction (PR) task loss with exponential decay for time gaps.

    Args:
        pr_numerical_outputs (list of Tensor): Predicted numerical values for PR task, one tensor per past event.
        pr_categorical_outputs (list of list of Tensor): Predicted categorical probabilities for PR task, one list of tensors per past event.
        target_numerical (Tensor): Ground truth numerical values for past events.
        target_categorical (list of Tensor): Ground truth categorical labels for past events.
        time_gaps (Tensor): Time differences between the current event and past events.
        lambda_param (float): Decay parameter for time gap weighting.
        max_past_events (int): Maximum number of past events to consider in the loss.
    
    Returns:
        Tensor: The PR task loss.
    """
    pr_loss_value = 0
    num_events = min(max_past_events, len(time_gaps))
    
    for k in range(num_events):
        # Calculate the exponential decay weight based on time gap
        weight = torch.exp(-time_gaps[k] / lambda_param)
        
        # MSE loss for numerical features at the k-th past event
        numerical_loss = F.mse_loss(pr_numerical_outputs[k], target_numerical[k], reduction='mean')
        
        # Cross-entropy loss for categorical features at the k-th past event
        categorical_loss = sum(F.cross_entropy(pred, target, reduction='mean') 
                               for pred, target in zip(pr_categorical_outputs[k], target_categorical[k])) / len(pr_categorical_outputs[k])
        
        # Weighted loss for the current past event
        pr_loss_value += weight * (numerical_loss + categorical_loss)
    
    return pr_loss_value


def nppr_loss(np_numerical_outputs, np_categorical_outputs, pr_numerical_outputs, pr_categorical_outputs, 
              target_numerical, target_categorical, time_gaps, lambda_param=1.0, max_past_events=5, alpha=0.5):
    """
    Combines NP and PR task losses with a weighting factor.

    Args:
        np_numerical_outputs (Tensor): Predicted numerical values for NP task.
        np_categorical_outputs (list of Tensor): Predicted categorical probabilities for NP task.
        pr_numerical_outputs (list of Tensor): Predicted numerical values for PR task.
        pr_categorical_outputs (list of Tensor): Predicted categorical probabilities for PR task.
        target_numerical (Tensor): Ground truth numerical values.
        target_categorical (list of Tensor): Ground truth categorical labels.
        time_gaps (Tensor): Timestamps for each event in the sequence.
        lambda_param (float): Decay parameter for PR task weights.
        max_past_events (int): Maximum number of past events to consider in the loss.
        alpha (float): Weight parameter balancing NP and PR tasks.
    
    Returns:
        Tensor: Combined NPPR loss.
    """
    # Compute NP loss
    np_loss_value = np_loss(np_numerical_outputs, np_categorical_outputs, target_numerical, target_categorical)
    
    # Compute PR loss
    pr_loss_value = pr_loss(pr_numerical_outputs, pr_categorical_outputs, target_numerical, target_categorical, time_gaps, lambda_param, max_past_events)
    
    # Weighted combination of NP and PR losses
    return (1 - alpha) * np_loss_value + alpha * pr_loss_value