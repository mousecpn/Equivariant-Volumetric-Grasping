import torch

import torch.nn.functional as F
    
def qual_loss_fn(pred, target, smoothing=0.0):
    # return F.binary_cross_entropy(pred, target, reduction="none")
    # return F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return binary_cross_entropy_with_label_smoothing(pred, target, reduction="none", smoothing=smoothing)


def width_loss_fn(pred, target):
    return F.mse_loss(40 * pred, 40 * target, reduction="none")

def occ_loss_fn(pred, target, smoothing=0.0):
    # return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)
    # return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(-1)
    return binary_cross_entropy_with_label_smoothing(pred, target, reduction="none", smoothing=smoothing).mean(-1)

def binary_cross_entropy_with_label_smoothing(logits, targets,reduction, smoothing=0.1):
    """
    计算带 Label Smoothing 的二元交叉熵损失
    :param logits: 模型的输出 (未经过 Sigmoid)
    :param targets: 真实标签 (0 or 1)
    :param smoothing: 平滑因子 (alpha), 通常取 0.1
    :return: BCE loss with label smoothing
    """
    # 平滑后的标签
    targets_smoothed = targets * (1 - smoothing) + 0.5 * smoothing
    
    # 计算 BCEWithLogitsLoss
    loss = F.binary_cross_entropy_with_logits(logits, targets_smoothed, reduction=reduction)
    
    return loss

def rot_loss_fn(pred, target):
    loss0 = quat_loss_fn(pred, target[:, 0])
    loss1 = quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)


def quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=-1))


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
    sigmoid: bool = True,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()  # (B, C)
    targets = targets.float()  # (B, C)
    
    if sigmoid is True:
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none") # (B, C)
    else:
        p = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_label_smoothing(inputs, targets, reduction="none", smoothing=smoothing) # (B, C)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)  # (B, C)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # # (B, C)
        loss = alpha_t * loss # (B, C)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss