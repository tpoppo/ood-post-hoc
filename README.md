# OOD post hoc methods
[![codecov](https://codecov.io/github/tpoppo/ood-post-hoc/branch/main/graph/badge.svg?token=3ORB55GD6O)](https://codecov.io/github/tpoppo/ood-post-hoc)

A TensorFlow implementation of various post hoc OOD detectors. All the methods should be compatible with the vast
majority of TensorFlow models.

## Usage
```python
import tensorflow as tf
from ood_detector.detectors import MSPDetector

model = tf.keras.applications.EfficientNetB0()
detector = MSPDetector(model)
ood_score = detector.predict(x_imgs)
```



## Methods

### MSP score [[1]](https://arxiv.org/abs/1610.02136)

The maximum softmax probability proposed in "Hendrycks and K. Gimpel. A baseline for detecting misclassified and
out-of-distribution examples in neural networks, 2016".

### ODIN [[2]](https://arxiv.org/abs/1706.02690)

The ODIN scoring function proposed in "S. Liang, Y. Li, and R. Srikant. Enhancing the reliability of out-of-distribution
image detection in neural networks, 2017". <br>

### Energy-based score [[3]](https://arxiv.org/abs/2010.03759)

The energy-based score proposed in "W. Liu, X. Wang, J. D. Owens, and Y. Li. Energy-based out-of-distribution detection,
2020".

### ReAct [[4]](https://arxiv.org/abs/2111.12797)

The ReAct layer proposed in "Y. Sun, C. Guo, and Y. Li. React: Out-of-distribution detection with rectified activations,
2021".

### GradNorm [[5]](https://arxiv.org/abs/2110.00218)

The GradNorm score proposed in "R. Huang, A. Geng, and Y. Li. On the importance of gradients for detecting distributional shifts in the wild".



## Project Structure

1) example.ipynb shows an example of how to use the various functions
2) evaluation notebook contains the notebooks used for the evaluation 
