import numpy as np
import scipy

from ood_detector.detectors.detector import Detector


class MSPDetector(Detector):
    """
    MSPDetector class implements the maximum softmax probability scoring function proposed by [1]
    [1] Hendrycks and K. Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks, 2016
    """

    def __init__(self, model, temperature=1000, args_predict=None):
        super().__init__(model, args_predict)
        self.temperature = temperature

    def predict(self, x, **kwargs):
        args_predict = self.args_predict.copy()
        args_predict.update(kwargs)

        y_pred = self.model.predict(x, **args_predict) / self.temperature
        y_pred = scipy.special.softmax(y_pred, axis=0)
        score = np.max(y_pred, axis=1)
        return score
