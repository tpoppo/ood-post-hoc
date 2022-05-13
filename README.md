# OOD post hoc methods
A TensorFlow implementation of various post hoc OOD detectors. All the methods should be compatible with the vast majority of TensorFlow models.

## Methods

### MSP score [[1]](https://arxiv.org/abs/1610.02136)
The maximum softmax probability proposed in "Hendrycks and K. Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks, 2016".


### ODIN [[2]](https://arxiv.org/abs/1706.02690)
The ODIN scoring function proposed in "S. Liang, Y. Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks, 2017". <br>
Note: This method might not work properly.

### energy-based score [[3]](https://arxiv.org/abs/2010.03759)
The energy-based score proposed in "W. Liu, X. Wang, J. D. Owens, and Y. Li. Energy-based out-of-distribution detection, 2020". 

### ReAct [[4]](https://arxiv.org/abs/2111.12797)
The ReAct layer proposed in "Y. Sun, C. Guo, and Y. Li. React: Out-of-distribution detection with rectified activations, 2021".

## Project Structure
1) example.ipynb shows an example of how to use the various functions
2) evaluation notebook contains the notebooks used for the evaluation 
