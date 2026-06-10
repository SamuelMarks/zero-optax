import numpy as np


def ctc_loss(
    logits, logit_paddings, labels, label_paddings, blank_id=0, log_epsilon=-1e5
):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        return np.zeros(np.array(getattr(logits, "data", logits)).shape[0])
    return logits


def ctc_loss_with_forward_probs(
    logits, logit_paddings, labels, label_paddings, blank_id=0, log_epsilon=-1e5
):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        B = l.shape[0]
        return np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
    return logits, logits, logits, logits
