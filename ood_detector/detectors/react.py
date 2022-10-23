import numpy as np
import tensorflow as tf

from ood_detector.detectors.energy import EnergyDetector
from ood_detector.detectors.msp import MSPDetector
from ood_detector.detectors.odin import OdinDetector


class ReactMSPDetector(MSPDetector):
    """
    The ReactMSPDetector class implements [1] using the scoring function from [2]

    [1] Y. Sun, C. Guo, and Y. Li. React: Out-of-distribution detection with rectified activations, 2021
    [2] Hendrycks and K. Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks, 2016
    """

    def __init__(self, model, x_id, *args, p=0.9, **kwargs):
        model = get_react_model(x_id, model, p=p)
        super().__init__(model, *args, **kwargs)


class ReactOdinDetector(OdinDetector):
    """
    The ReactOdinDetector class implements [1] using the scoring function from [2]

    [1] Y. Sun, C. Guo, and Y. Li. React: Out-of-distribution detection with rectified activations, 2021
    [2] S. Liang, Y. Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks, 2017
    """

    def __init__(self, model, x_id, *args, p=0.9, **kwargs):
        model = get_react_model(x_id, model, p=p)
        super().__init__(model, *args, **kwargs)


class ReactEnergyDetector(EnergyDetector):
    """
    The ReactEnergyDetector class implements [1] using the scoring function from [2]

    [1] Y. Sun, C. Guo, and Y. Li. React: Out-of-distribution detection with rectified activations, 2021
    [2] W. Liu, X. Wang, J. D. Owens, and Y. Li. Energy-based out-of-distribution detection, 2020
    """

    def __init__(self, model, x_id, *args, p=0.9, **kwargs):
        model = get_react_model(x_id, model, p=p)
        super().__init__(model, *args, **kwargs)


def get_react_model(x_id, model_base, p=0.9, show_info=False):
    pos = -1
    while True:
        layer_fv = model_base.get_layer(index=pos)
        if layer_fv.output.shape[-1] != model_base.get_layer(index=-1).output.shape[-1]:
            break
        pos -= 1

    model_fv = tf.keras.Model(inputs=model_base.input, outputs=layer_fv.output)
    model_fv = tf.keras.Sequential([model_fv, tf.keras.layers.Flatten()])
    y = model_fv.predict(x_id).flatten()
    y_pos = min(int(p * len(y)), len(y) - 1)
    lambda_val = np.partition(y, pos, axis=0)[y_pos]
    if show_info:
        print(f"lambda_val: {lambda_val}")

    inp = tf.keras.layers.Input(shape=model_fv.output.shape[1:])
    x = inp
    for p in range(pos + 1, 0):
        layer = model_base.get_layer(index=p)
        try:
            if layer.activation == tf.keras.activations.softmax:
                layer.activation = tf.keras.activations.linear
                if show_info:
                    print("softmax layer removed")
        except Exception as e:
            if show_info:
                print(e)
        x = layer(x)

    last_block = tf.keras.Model(inputs=inp, outputs=x)
    last_block.compile("adam")

    # feature vector
    model_react = tf.keras.Sequential([model_fv, ReAct(lambda_val), last_block])

    model_react.compile("adam")
    return model_react


class ReAct(tf.keras.layers.Layer):
    """
    The ReAct class implements the ReAct layer from [1]

    [1] Y. Sun, C. Guo, and Y. Li. React: Out-of-distribution detection with rectified activations, 2021
    """

    def __init__(self, lambda_val):
        super().__init__()
        self.lambda_val = lambda_val

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.math.minimum(inputs, self.lambda_val)
