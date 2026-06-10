import numpy as np


def hinge_loss(predictor_outputs, targets):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(targets, "data", targets))
        return np.maximum(0.0, 1.0 - p * t)
    return predictor_outputs


def perceptron_loss(predictor_outputs, targets):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(targets, "data", targets))
        return np.maximum(0.0, -p * t)
    return predictor_outputs


def safe_softmax_cross_entropy(logits, labels, where=None):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        c = np.max(l, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(l - c), axis=-1, keepdims=True))
        log_softmax = l - c - logsumexp
        res = -np.sum(t * log_softmax, axis=-1)
        if where is not None:
            w = np.array(getattr(where, "data", where))
            res = np.where(w, res, 0.0)
        return res
    return logits


def softmax_cross_entropy(logits, labels, where=None):
    return safe_softmax_cross_entropy(logits, labels, where=where)


def softmax_cross_entropy_with_integer_labels(logits, labels, where=None):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        c = np.max(l, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(l - c), axis=-1, keepdims=True))
        log_softmax = l - c - logsumexp
        res = -np.take_along_axis(log_softmax, np.expand_dims(t, -1), axis=-1).squeeze(
            -1
        )
        if where is not None:
            w = np.array(getattr(where, "data", where))
            res = np.where(w, res, 0.0)
        return res
    return logits


def sigmoid_binary_cross_entropy(logits, labels):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        return np.maximum(l, 0) - l * t + np.log1p(np.exp(-np.abs(l)))
    return logits


def sigmoid_focal_loss(logits, labels, alpha=None, gamma=2.0):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        p = 1 / (1 + np.exp(-l))
        ce = np.maximum(l, 0) - l * t + np.log1p(np.exp(-np.abs(l)))
        p_t = p * t + (1 - p) * (1 - t)
        loss = ce * ((1 - p_t) ** gamma)
        if alpha is not None:
            alpha_t = alpha * t + (1 - alpha) * (1 - t)
            loss = alpha_t * loss
        return loss
    return logits


def multiclass_hinge_loss(predictor_outputs, labels):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(labels, "data", labels))
        correct_p = np.take_along_axis(p, np.expand_dims(t, -1), axis=-1)
        margins = np.maximum(0.0, 1.0 - correct_p + p)
        np.put_along_axis(margins, np.expand_dims(t, -1), 0.0, axis=-1)
        return np.sum(margins, axis=-1)
    return predictor_outputs


def multiclass_perceptron_loss(predictor_outputs, labels):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(labels, "data", labels))
        correct_p = np.take_along_axis(p, np.expand_dims(t, -1), axis=-1)
        margins = np.maximum(0.0, -correct_p + p)
        np.put_along_axis(margins, np.expand_dims(t, -1), 0.0, axis=-1)
        return np.sum(margins, axis=-1)
    return predictor_outputs


def poly_loss_cross_entropy(logits, labels, epsilon=2.0, where=None):
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        ce = safe_softmax_cross_entropy(l, t)
        c = np.max(l, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(l - c), axis=-1, keepdims=True))
        p = np.exp(l - c - logsumexp)
        pt = np.sum(p * t, axis=-1)
        res = ce + epsilon * (1 - pt)
        if where is not None:
            w = np.array(getattr(where, "data", where))
            res = np.where(w, res, 0.0)
        return res
    return logits
