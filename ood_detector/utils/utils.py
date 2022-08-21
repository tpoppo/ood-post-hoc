import numpy as np
import tensorflow as tf
from ood_metrics import auroc, fpr_at_95_tpr


def get_batch_gradient(model, loss_object, x, y_true=None):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        if y_true is None:
            loss = loss_object(y_pred)
        else:
            loss = loss_object(y_true, y_pred)
    gradient = tape.gradient(loss, x)
    del loss, loss_object, x
    return gradient.numpy()


def get_gradient(model, loss_object, x, batch_size, y_true=None):
    grads = []
    pos = 0
    while pos < len(x):
        if y_true is not None:
            y_true_batch = y_true[pos:pos + batch_size]
        else:
            y_true_batch = None
        grads.append(get_batch_gradient(model, loss_object, x[pos:pos + batch_size], y_true=y_true_batch))
        pos += batch_size
    return np.concatenate(grads, axis=0)


def evaluate_aucroc(detector, x_id, x_ood):
    score_id = detector.predict(x_id)
    score_ood = detector.predict(x_ood)
    y_pred = np.array([1] * len(score_id) + [0] * len(score_ood))

    return auroc(np.concatenate([score_id, score_ood], axis=0), y_pred)


def evaluate_fpr_at_95_tpr(detector, x_id, x_ood):
    score_id = detector.predict(x_id)
    score_ood = detector.predict(x_ood)
    y_pred = np.array([1] * len(score_id) + [0] * len(score_ood))

    return fpr_at_95_tpr(np.concatenate([score_id, score_ood], axis=0), y_pred)
