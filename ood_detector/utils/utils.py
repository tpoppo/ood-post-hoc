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


def get_batch_layer_gradient(model, loss_object, layer, x, y_true=None):
    x = tf.convert_to_tensor(x)

    with tf.GradientTape() as tape:
        weights = (
            layer.kernel
        )  # tf.reshape(layer.kernel, [-1]) # tf.concat([tf.reshape(w, [-1]) for w in layer.weights], axis=0)
        tape.watch(weights)
        y_pred = model(x)
        if y_true is None:
            loss = loss_object(y_pred)
        else:
            loss = loss_object(y_true, y_pred)

    gradient = tape.jacobian(loss, weights)
    return gradient.numpy().reshape(x.shape[0], -1)


def get_gradient(model, loss_object, x, batch_size, y_true=None):
    """
    Get the gradient of the model with respect to the input using the given loss function.
    :param model: the TensorFlow model
    :param loss_object: a python function for the loss. It must be loss(y) or loss(y_true, y_pred)
    :param x: input data
    :param batch_size: batch size
    :param y_true: y_true if the loss wants loss(y_true, y_pred), otherwise is None.
    :return: The gradient of the model w.r.t. the input
    """
    grads = []
    pos = 0
    while pos < len(x):
        if y_true is not None:
            y_true_batch = y_true[pos : pos + batch_size]
        else:
            y_true_batch = None
        grads.append(
            get_batch_gradient(
                model, loss_object, x[pos : pos + batch_size], y_true=y_true_batch
            )
        )
        pos += batch_size
    return np.concatenate(grads, axis=0)


def get_layer_gradient(model, loss_object, x, batch_size, layer, y_true=None):
    """
    Get the gradient of the model with respect to the input using the given loss function.
    :param model: the TensorFlow model
    :param loss_object: a python function for the loss. It must be loss(y) or loss(y_true, y_pred)
    :param x: input data
    :param batch_size: batch size
    :param y_true: y_true if the loss wants loss(y_true, y_pred), otherwise is None.
    :param layer: layer used to obtain the gradient
    :return: The gradient of the model w.r.t. the layer's weights
    """
    grads = []
    pos = 0
    while pos < len(x):
        if y_true is not None:
            y_true_batch = y_true[pos : pos + batch_size]
        else:
            y_true_batch = None
        grads.append(
            get_batch_layer_gradient(
                model,
                loss_object,
                layer,
                x[pos : pos + batch_size],
                y_true=y_true_batch,
            )
        )
        pos += batch_size
    return np.concatenate(grads, axis=0)


def evaluate_aucroc(detector, x_id, x_ood):
    """
    It returns the aucroc of the detector
    :param detector: Detector class
    :param x_id: in distribution data
    :param x_ood:  out of distribution data
    :return: auc roc score
    """
    score_id = detector.predict(x_id)
    score_ood = detector.predict(x_ood)
    y_pred = np.array([1] * len(score_id) + [0] * len(score_ood))

    return auroc(np.concatenate([score_id, score_ood], axis=0), y_pred)


def evaluate_fpr_at_95_tpr(detector, x_id, x_ood):
    """
    It returns the false positive ratio at 95 true positive ratio of the detector
    :param detector: Detector class
    :param x_id: in distribution data
    :param x_ood:  out of distribution data
    :return: fpr at 95 tpr score
    """

    score_id = detector.predict(x_id)
    score_ood = detector.predict(x_ood)
    y_pred = np.array([1] * len(score_id) + [0] * len(score_ood))

    return fpr_at_95_tpr(np.concatenate([score_id, score_ood], axis=0), y_pred)
