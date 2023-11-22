# Staircase Boolean Functions

Code adapted from "The staircase property: How hierarchial structure can guide deep learning" (NeurIPS 2021).

- `neural_net_architectures.py`: Contains neural network architectures used in training.
- `utils.py`: Contains utilities for working with Boolean functions and their Fourier coefficients.
- `datasets.py`: Creates datasets according to probability distributions of data.
- `train.py`: Contains code for training network, storing their weights and losses.
- `staircase_resnet_tests[*].py`: Runs code for various settings:
  - `None`: Fits a ResNet to a staircase function / parity function evaluated over the unbiased hypercube
  - `_biased`: Fits a ResNet to a staircase function / parity function evaluated over the biased hypercube (with parameter `p`)
  - `_gauss`: Fits a ResNet to a staircase function / parity function evaluated over a standard Gaussian
  - `_multi`: Fits a ResNet to a multi-staircase function (of degree `d_1` and `d_2`) evaluated over the unbiased hypercube