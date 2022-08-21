import numpy as np
import tensorflow as tf

from ood_detector.detectors.energy import EnergyDetector
from ood_detector.detectors.msp import MSPDetector
from ood_detector.detectors.odin import OdinDetector


class ReactMSPDetector(MSPDetector):
    def __init__(self, model, x_id, p=0.9, *args, **kwargs):
        model = get_react_model(x_id, model, p=p)
        super().__init__(model, *args, **kwargs)


class ReactOdinDetector(OdinDetector):
    def __init__(self, model, x_id, p=0.9, *args, **kwargs):
        model = get_react_model(x_id, model, p=p)
        super().__init__(model, *args, **kwargs)


class ReactEnergyDetector(EnergyDetector):
    def __init__(self, model, x_id, p=0.9, *args, **kwargs):
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
    λ = np.partition(y, pos, axis=0)[y_pos]
    if show_info:
        print(f'λ: {λ}')

    inp = tf.keras.layers.Input(shape=model_fv.output.shape[1:])
    x = inp
    for p in range(pos + 1, 0):
        layer = model_base.get_layer(index=p)
        try:
            if layer.activation == tf.keras.activations.softmax:
                layer.activation = tf.keras.activations.linear
                if show_info:
                    print('softmax layer removed')
        except Exception as e:
            if show_info:
                print(e)
        x = layer(x)

    last_block = tf.keras.Model(inputs=inp, outputs=x)
    last_block.compile('adam')

    # feature vector
    model_react = tf.keras.Sequential([
        model_fv,
        ReAct(λ),
        last_block
    ])

    model_react.compile('adam')
    return model_react


class ReAct(tf.keras.layers.Layer):
    def __init__(self, λ):
        super(ReAct, self).__init__()
        self.λ = λ

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.math.minimum(inputs, self.λ)
