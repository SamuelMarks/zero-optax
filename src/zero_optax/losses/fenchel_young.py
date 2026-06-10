import numpy as np


def make_fenchel_young_loss(max_prob_fn, *args, **kwargs):
    def loss(logits, labels, **kwargs):
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            l = np.array(getattr(logits, "data", logits))
            return np.zeros(l.shape)
        return logits

    return loss
