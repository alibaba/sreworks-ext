import torch


def cross_entropy(proba_pred, proba_target):
    """Computes the cross entropy between the two probabilities torch tensors."""
    # proba_pred = proba_pred+1e-10
    return -(proba_target * torch.log(proba_pred)).mean()


def cross_entropy_multiple(proba_pred, proba_target):
    return -(proba_target * torch.log(proba_pred)).mean(dim=[0, 2, 3])


def log_loss(proba_pred, proba_target):
    """Computes the log loss between the two probabilities torch tensors."""
    # proba_pred = proba_pred+1e-10
    label_target = torch.argmax(proba_target, dim=-1, keepdim=True)
    proba_select = torch.gather(proba_pred, -1, label_target)
    return -(torch.log(proba_select)).mean()


def log_loss_multiple(proba_pred, proba_target):
    label_target = torch.argmax(proba_target, dim=-1, keepdim=True)
    proba_select = torch.gather(proba_pred, -1, label_target)
    logloss = -torch.mean(torch.log(proba_select), dim=[0, 2, 3])
    return logloss


def log_loss_target(proba_pred, target):
    """Computes log loss between the target and the predicted probabilities expressed as torch tensors.

    The target is a one dimensional tensor whose dimension matches the first dimension of proba_pred.
    It contains integers that represent the true class for each instance.
    """
    proba_select = torch.gather(proba_pred, -1, target)
    return -(torch.log(proba_select)).mean()


def mse(Y, Y_target):
    """Computes the mean squared error between Y and Y_target."""
    return torch.mean((Y - Y_target) ** 2)


def mse_multiple(Y, Y_target):
    """Computes the mean squared error between Y and Y_target."""
    return torch.mean((Y - Y_target) ** 2, dim=[0, 2, 3])