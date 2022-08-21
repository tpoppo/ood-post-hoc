import numpy as np
import scipy

from ood_detector.detectors.detector import Detector


class MSPDetector(Detector):

    def __init__(self, model, t=1000, args_predict=None):
        super().__init__(model, args_predict)
        self.t = t

    def predict(self, x, **kwargs):
        args_predict = self.args_predict.copy()
        args_predict.update(kwargs)

        y_pred = self.model.predict(x, **args_predict) / self.t
        y_pred = scipy.special.softmax(y_pred, axis=0)
        score = np.max(y_pred, axis=1)
        return score
