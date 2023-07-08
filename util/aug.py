
from util.sc import *
import random
import numpy as np
import torch
from torchlibrosa.augmentation import SpecAugmentation
import numpy as np


def core_mixup(X, y, alpha=0.2, beta=0.2):
    indices = torch.randperm(X.shape[0])

    X2 = X[indices, :, :, :]
    y2 = y[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, beta)])
    X = lam * X + (1 - lam) * X2
    y = lam * y + (1 - lam) * y2

    return X, y


def mixup(X, y):
    X, y = core_mixup(X, y)
    return X, y


# SpecAugment
spec_augmenter = SpecAugmentation(
    time_drop_width=2,
    time_stripes_num=2,
    freq_drop_width=2,
    freq_stripes_num=2)
