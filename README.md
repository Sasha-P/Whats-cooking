**Introduction**

The repo contains different implementations of the Kaggle Whats-cooking challenge (<https://www.kaggle.com/c/whats-cooking>)

**Installation and Configuration**

Install Anaconda Open Data Science Platform (Linux)

1. Download and Install Anaconda

    Download Anaconda distribution from <https://www.continuum.io/downloads>

    Run `Anaconda3-4.0.0-Linux-x86_64.sh` to install (without `sudo`)

2. Create a virtual environment

    Run `conda create --name <env_name> --file </path/to/file/>requirements.txt`
    Type “y” for “yes.”

3. Change environments (activate)

    `source activate <env_name>`

For more information about Anaconda, visit <https://www.continuum.io/>

`conda` documentation can be found at <http://conda.pydata.org/docs/index.html>

**Applications**

* `main_nn.py` and `funcsEx04.py` - A neural network based on Machine Learning course on Coursera <https://www.coursera.org/learn/machine-learning>.
Rewritten in Python by royshoo (<https://github.com/royshoo/mlsn>) with my modification for the Whats-cooking task.
* `main_sklearn_svm.py` - OneVsRestClassifier on SVM core with the scikit-learn framework (http://scikit-learn.org/).
* `main_sklearn_lr.py` - OneVsRestClassifier on Linear Regression core.
* `main_tf_gd.py` - !Experimental! Tensorflow framework gradient descent (<https://www.tensorflow.org/>).
* `utils.py` - utils methods for work with the Whats-cooking dataset.

All rights belong to their owners.