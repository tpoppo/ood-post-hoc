import unittest

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import ood_detector.detectors as detectors
import ood_detector.utils as utils

N_IMAGES = 32
BATCH_SIZE = 16


def evaluate_detector(self, detector, verbose=True):
    score_aucroc = utils.evaluate_aucroc(detector, self.x_id, self.x_ood)
    score_fpr_at_95_tpr = utils.evaluate_fpr_at_95_tpr(detector, self.x_id, self.x_ood)

    if verbose:
        print(score_aucroc, score_fpr_at_95_tpr)

    self.assertGreaterEqual(score_aucroc, 0.5)
    self.assertGreaterEqual(score_fpr_at_95_tpr, 0.1)


class TestImageNetGaussianEfficientNetB0(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = tf.keras.applications.EfficientNetB0()

        # ood dataset
        self.x_ood = np.random.normal(loc=0.5, scale=0.2, size=(N_IMAGES, 224, 224, 3))

        # id dataset
        ds_id = tfds.load(
            "imagenette/160px-v2",
            split=f"validation[:{N_IMAGES}]",
            as_supervised=True,
            try_gcs=True,
        )
        ds_id = ds_id.map(
            lambda x, y: (
                tf.image.resize(tf.image.convert_image_dtype(x, tf.float32), (224, 224))
            )
        )  # [0, 1]
        ds_id = ds_id.map(tf.keras.applications.mobilenet_v3.preprocess_input)
        ds_id = ds_id.batch(BATCH_SIZE)
        self.x_id = np.concatenate([a.numpy() for a in ds_id], axis=0)

    def test_msp(self):
        """
        Test for the MSPDetector class
        """
        detector = detectors.MSPDetector(
            self.model, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)

    def test_energy(self):
        """
        Test for the EnergyDetector class
        """

        detector = detectors.EnergyDetector(
            self.model, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)

    def test_odin(self):
        """
        Test for the OdinDetector class
        """
        detector = detectors.OdinDetector(
            self.model, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)

    def test_react_msp(self):
        """
        Test for the ReactMSPDetector class
        """
        detector = detectors.ReactMSPDetector(
            self.model, self.x_id, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)

    def test_react_odin(self):
        """
        Test for the ReactOdinDetector class
        """
        detector = detectors.ReactOdinDetector(
            self.model, self.x_id, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)

    def test_react_energy(self):
        detector = detectors.ReactEnergyDetector(
            self.model, self.x_id, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)

    def test_gradnorm(self):
        detector = detectors.GradNormDetector(
            self.model, args_predict={"batch_size": BATCH_SIZE}
        )
        evaluate_detector(self, detector)


if __name__ == "__main__":
    unittest.main()
