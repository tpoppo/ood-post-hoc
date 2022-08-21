import numpy as np
import scipy
import tensorflow as tf

from ood_detector.detectors.detector import Detector
from ..utils.utils import get_gradient


class OdinDetector(Detector):

    def __init__(self, model, epsilon=0.0034, t=1000, args_predict=None):
        super().__init__(model, args_predict)
        self.epsilon = epsilon
        self.t = t

    def predict(self, x, **kwargs):
        args_predict = self.args_predict.copy()
        args_predict.update(kwargs)

        batch_size = args_predict.get('batch_size', 32)

        x = odin_perturbation(self.model, x, t=self.t, epsilon=self.epsilon, batch_size=batch_size)
        y_pred = self.model.predict(x, **args_predict) / self.t
        y_pred = scipy.special.softmax(y_pred, axis=0)
        y_pred = np.max(y_pred, axis=1)
        return y_pred


def odin_perturbation(model, x, t, epsilon, batch_size):
    def loss(y_pred):
        y_pred = y_pred / t
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        return tf.math.log(tf.math.reduce_max(y_pred, axis=0))

    g = get_gradient(model, loss, x.copy(), batch_size)
    x -= epsilon * np.sign(-g)
    return x
