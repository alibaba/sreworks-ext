import numpy as np
import torch

from utils.tensor_manipulation import extract_subtensor


def get_information(
    saliency: torch.Tensor, ids_time=None, ids_feature=None, normalize: bool = False, eps: float = 1.0e-5
):

    subsaliency = extract_subtensor(saliency, ids_time, ids_feature)
    subsaliency_information = (torch.abs(torch.log2(1 - subsaliency + eps))).sum()
    if normalize:
        saliency_information = (torch.abs(torch.log2(1 - subsaliency + eps))).sum()
        subsaliency_information /= saliency_information
    return subsaliency_information.cpu().item()


def get_entropy(saliency: torch.Tensor, ids_time=None, ids_feature=None, normalize: bool = False, eps: float = 1.0e-5):

    subsaliency = extract_subtensor(saliency, ids_time, ids_feature)
    subentropy_tensor = subsaliency * torch.abs(torch.log2(eps + subsaliency)) + (1 - subsaliency) * torch.abs(
        torch.log2(eps + 1 - subsaliency)
    )
    subsaliency_entropy = subentropy_tensor.sum()
    if normalize:
        entropy_tensor = saliency * torch.abs(torch.log2(eps + saliency)) + (1 - saliency) * torch.abs(
            torch.log2(eps + 1 - saliency)
        )

        saliency_entropy = entropy_tensor.sum()
        subsaliency_entropy /= saliency_entropy
    return subsaliency_entropy.cpu().item()


def get_information_array(saliency: np.ndarray, eps: float = 1.0e-5):

    saliency_information = (np.abs(np.log2(1 - saliency + eps))).sum()
    return saliency_information


def get_entropy_array(saliency: np.ndarray, eps: float = 1.0e-5):

    entropy_tensor = saliency * np.abs(np.log2(eps + saliency)) + (1 - saliency) * np.abs(np.log2(eps + 1 - saliency))
    saliency_entropy = entropy_tensor.sum()
    return saliency_entropy
