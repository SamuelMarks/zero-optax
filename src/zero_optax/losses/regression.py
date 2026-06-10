import numpy as np


def squared_error(predictions, targets=None):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictions, "data", predictions))
        if targets is None:
            t = np.zeros_like(p)
        else:
            t = np.array(getattr(targets, "data", targets))
        return (p - t) ** 2
    return predictions


def l2_loss(predictions, targets=None):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictions, "data", predictions))
        if targets is None:
            t = np.zeros_like(p)
        else:
            t = np.array(getattr(targets, "data", targets))
        return 0.5 * (p - t) ** 2
    return predictions


def huber_loss(predictions, targets=None, delta=1.0):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictions, "data", predictions))
        if targets is None:
            t = np.zeros_like(p)
        else:
            t = np.array(getattr(targets, "data", targets))
        diff = np.abs(p - t)
        return np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return predictions
