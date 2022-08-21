from abc import ABC, abstractmethod


class Detector(ABC):

    def __init__(self, model, args_predict=None):
        if args_predict is None:
            args_predict = {}
        self.model = model
        self.args_predict = args_predict

    @abstractmethod
    def predict(self, x):
        pass
