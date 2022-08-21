import numpy as np

from ood_detector.detectors.detector import Detector


class EnergyDetector(Detector):

    def __init__(self, model, epsilon=1e-6, args_predict=None):
        super().__init__(model, args_predict)
        self.epsilon = epsilon

    def predict(self, x, **kwargs):
        args_predict = self.args_predict.copy()
        args_predict.update(kwargs)

        y_pred = self.model.predict(x, **args_predict)
        score = np.log(self.epsilon + np.sum(np.exp(y_pred), axis=1))
        return score
