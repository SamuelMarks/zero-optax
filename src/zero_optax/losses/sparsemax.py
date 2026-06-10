import numpy as np


def sparsemax_loss(logits, labels):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        return np.zeros(l.shape[:-1])
    return logits


def multiclass_sparsemax_loss(logits, labels):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        return np.zeros(l.shape[:-1])
    return logits
