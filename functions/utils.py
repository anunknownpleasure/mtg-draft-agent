import numpy as np
import torch


def ceil_digit(x, digits=2):
    """
    Rounds up at a specified decimal place.
    E.g. ceil_digits(1.234, digits=2) == 1.24
    """
    return np.ceil(x * 10**digits) / 10**digits


def create_logit_mask(options_batch, vocab_size, padding_idx=None):
    """
    Turns a list of option tokens into a boolean mask.
    """
    batch_size = options_batch.shape[0]
    device = options_batch.device

    mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)

    for idb in range(batch_size):
        options = options_batch[idb, :]

        # Only create a mask for non-padding indices
        if padding_idx is not None:
            I = options != padding_idx
            options = options[I]

        mask[idb, options] = True

    return mask
