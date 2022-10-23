import tensorflow as tf
import numpy as np

from ood_detector.detectors.detector import Detector
from ..utils.utils import get_layer_gradient


class GradNormDetector(Detector):
    """
    The GradNormDetector class implements the energy-based scoring function proposed in [1].

    [1] R. Huang, A. Geng, and Y. Li. On the importance of gradients for detecting distributional shifts in the wild.
    """

    def __init__(self, model, temperature=1.0, args_predict=None):
        super().__init__(model, args_predict)
        self.temperature = temperature

        pos_layer = -1
        while True:
            self.layer = self.model.get_layer(index=pos_layer)
            if self.layer.get_weights():
                break
            pos_layer -= 1

    def predict(self, x, **kwargs):
        args_predict = self.args_predict.copy()
        args_predict.update(kwargs)

        batch_size = args_predict.get("batch_size", 32)

        @tf.function(
            input_signature=(
                tf.TensorSpec(
                    shape=[None, self.model.output.shape[-1]],
                    dtype=self.model.output.dtype,
                ),
            )
        )
        def loss_object(logits):
            return tf.math.reduce_mean(
                tf.nn.log_softmax(logits / self.temperature, axis=-1), axis=-1
            )

        gradient = get_layer_gradient(
            self.model, loss_object, x, batch_size, self.layer, y_true=None
        )
        return np.abs(gradient).sum(axis=-1)
