import numpy as np


def ranking_softmax_loss(logits, labels, weights=None, where=None, reduce_fn=np.mean):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        res = np.zeros(l.shape[:-1])
        if reduce_fn is None:
            return res
        return reduce_fn(res)
    return logits
