# DAGMM Tensorflow implementation
Deep Autoencoding Gaussian Mixture Model.

This implementation is based on the paper
**Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection**
[[Bo Zong et al (2018)]](https://openreview.net/pdf?id=BJJLHbb0-)

Source code is from https://github.com/tnakae/DAGMM

# Requirements
- python (3.5-3.6)
- Tensorflow <= 1.15
- Numpy
- sklearn
- arch == 4.15
- matplotlib
- scipy
- statsmodels
- tsfresh
- tqdm

# Usage instructions
To use DAGMM model, you need to create "DAGMM" object.
At initialize, you have to specify next 4 variables at least.

- ``comp_hiddens`` : list of int
  - sizes of hidden layers of compression network
  - For example, if the sizes are ``[n1, n2]``,
  structure of compression network is:
  ``input_size -> n1 -> n2 -> n1 -> input_sizes``
- ``comp_activation`` : function
  - activation function of compression network
- ``est_hiddens`` : list of int
  - sizes of hidden layers of estimation network.
  - The last element of this list is assigned as n_comp.
  - For example, if the sizes are ``[n1, n2]``,
    structure of estimation network is:
    ``input_size -> n1 -> n2 (= n_comp)``
- ``est_activation`` : function
  - activation function of estimation network

Then you fit the training data, and predict to get energies
(anomaly score). It looks like the model interface of scikit-learn.

For more details, please check out [dagmm/dagmm.py](dagmm/dagmm.py) docstrings.

## Usage guide
To get the result of public datasets and holo datasets using DAGMM, you need to define the location of the datasets which has already been processed.

In [main.py](main.py):

```python
#datasets folder
public_datafolder = ''
holo_datafolder = ''
holo_datasets = os.listdir(holo_datafolder)
holo_result_file = 'holo_result.csv'
pub_result_file = 'pub_result.csv'
#SPOT config
q = 1e-4
lm = 0.999
```
After setting all config properly, you can run the model in terminal.
```bash
python3 main.py
```

# Notes
## GMM Implementation
The equation to calculate "energy" for each sample in the original paper
uses direct expression of multivariate gaussian distribution which
has covariance matrix inversion, but it is impossible sometimes
because of singularity.

Instead, this implementation uses cholesky decomposition of covariance matrix.
(this is based on [GMM code in Tensorflow code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/factorization/python/ops/gmm_ops.py))

In ``DAGMM.fit()``, it generates and stores triangular matrix of cholesky decomposition
of covariance matrix, and it is used in ``DAGMM.predict()``,

In addition to it, small perturbation (1e-6) is added to diagonal
elements of covariance matrix for more numerical stability
(it is same as Tensorflow GMM implementation,
and [another author of DAGMM](https://github.com/danieltan07/dagmm) also points it out)

## Parameter of GMM Covariance (lambda_2)
Default value of lambda_2 is set to 0.0001 (0.005 in original paper).
When lambda_2 is 0.005, covariances of GMM becomes too large to detect
anomaly points. But perhaps it depends on distribution of data and method of preprocessing
(for example a method of normalization). Recommend to control lambda_2
when performance metrics is not good.
