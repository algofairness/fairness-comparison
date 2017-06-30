# Black Box Auditing and Certifying and Removing Disparate Impact

This repository contains a sample implementation of Gradient Feature Auditing (GFA) meant to be generalizable to most datasets.  For more information on the repair process, see our paper on [Certifying and Removing Disparate Impact](http://arxiv.org/abs/1412.3756).  For information on the full auditing process, see our paper on [Auditing Black-box Models for Indirect Influence](http://arxiv.org/abs/1602.07043).

# License

This code is licensed under an [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html) license.

# Certifying and Removing Disparate Impact

If all you want to do is run the data repair described in [Certifying and Removing Disparate Impact](http://arxiv.org/abs/1412.3756), you should be able to do that without the installation steps described below.  To repair data with respect to a single (e.g., protected) attribute, use the `repair.py` file.  Running `python repair.py` will tell you the arguments the script takes.

# Black Box Auditing

To run GFA on a dataset (as in [Auditing Black-box Models for Indirect Influence](http://arxiv.org/abs/1602.07043)), use the `main.py` file. The top few lines of that file dictate what machine-learning technique is to be used (the "model factory"), what dataset should be loaded (the "experiment"), and what the response-feature of the data-set is. You also may specify certain dataset features to ignore in the training/auditing process, as well as which "measurers" you would like to use for GFA.

## Creating a New "Experiment" / Using a New Dataset

Each "Experiment" should reside in the `experiments` directory as a separate module; each such module should have a load_data method prescribed in the `__init__.py` file (refer to `experiments/sample/__init__.py` for an example). This `load_data` method should return a tuple containing (in order) the headers, training set, and test set for the experiment.

## Testing Code Changes

All tests should be run from the main project directory. To make this process easier, a `run_test_suite.sh` file has been included (which can be run with bash via: `bash ./run_test_suite.sh`) in order to run all available tests at once.

Every python file should include test functions at the bottom that will be run when the file is run. This can be done by including the line `if __name__=="__main__": test()` as long as there is a function defined as `test`.

These tests should use print statements with `True` or `False` readouts indicating success or failure (where `True` should always be success). It is fine/good to have multiple of these per file.

Note: if a test requires reading data from the `test_data` directory, it should import the appropriate `load_data` file from the `experiments` directory.

## Implementing a New Machine-Learning Method

The best way to create a model would be to use a ModelFactory and ModelVisitors. A ModelVisitor should be thought of as a wrapper that knows how to load a machine-learning model of a given type and communicate with that model file in order to output predicted values of some test dataset. A ModelFactory simply knows how to "build" a ModelVisitor based on some provided training data. Check out the "Abstract" files in the `sample_experiment` directory for outlines of what these two classes should do; similarly, check out the "SVM_ModelFactory" files in the `sample_experiment` subdirectory for examples that use WEKA to create model files and produce predictions.

## Setup and Installation

1. Clone this repository to your workspace.
2. Install WEKA and/or Tensorflow (see below).
3. Update the WEKA path in `model_factories/AbstractWekaModelFactory.py`.
4. Install the Python dependencies listed in the requirements.txt file.
5. Install python-matplotlib if you do not already have it (`sudo apt-get install python-matplotlib`).
6. Run `python main.py` to run the sample experiment.

Many of the ModelVisitors rely on [Weka](http://www.cs.waikato.ac.nz/ml/weka/). Similarly, we use [TensorFlow](https://www.tensorflow.org/) for network-based machine learning. Any Python libraries that need to be installed are included in the `requirements.txt` file.
- Weka 3.6.13 [download](http://www.cs.waikato.ac.nz/ml/weka/downloading.html)
- TensorFlow [download](https://www.tensorflow.org/versions/master/get_started/os_setup.html) (original experiments run with version 0.6.0)

# Sources

Dataset Sources:
 - adult.csv [link](https://archive.ics.uci.edu/ml/datasets/Adult)
 - german_categorical.csv (Modified from [link](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
 - RicciDataMod.csv (Modified from [link](http://www.amstat.org/publications/jse/v18n3/RicciData.csv))
 - DRP Datasets (Source and data-files coming soon.)
 - Arrests/Recidivism Datasets [link](http://www.icpsr.umich.edu/icpsrweb/RCMD/studies/3355)
 - Linear Datasets ("sample_2" Experiment) [link](https://github.com/jasonbaldridge/try-tf)

More information on DRP can be found at the [Dark Reactions Project](http://darkreactions.haverford.edu/) official site.

# Bug Reports and Feature-Requests

All bug reports and feature-requests should be submitted through the [Issue Tracker](https://github.com/cfalk/BlackBoxAuditing/issues).
