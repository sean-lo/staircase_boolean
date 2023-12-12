# Staircase Boolean Functions

Code adapted from 
["The staircase property: How hierarchial structure can guide deep learning"](https://arxiv.org/abs/2108.10573).

- `experiments/`: sub-directory containing files for the experiments run. 
  - The experiment `experiment.py` is run on a cloud computing cluster, reading args from `data.csv`.
  - `postprocessing.py.ipynb` performs post-experiment processing. Run this if you only want to recreate the figures.
- `staircase/`: contains the source code for staircase function generation and training.
  - `neural_net_architectures.py`: Contains neural network architectures used in training.
  - `utils.py`: Contains utilities for working with Boolean functions and their Fourier coefficients.
  - `datasets.py`: Creates datasets according to probability distributions of data.
  - `train.py`: Contains code for training network, storing their weights and losses.
- `tests/staircase_custom_resnet_tests.py`: Tests the training setup.