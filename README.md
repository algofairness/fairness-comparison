This repository is meant to facilitate the benchmarking of fairness aware machine learning algorithms.

To run the benchmarks, clone the repository and run:
> python3 benchmark.py

This will write out metrics for each dataset to the results/ directory.

*Optional*:  The benchmarks rely on preprocessed versions of the datasets that have been included
in the repository.  If you would like to regenerate this preprocessing, run the below command
before running the benchmark script:
> python3 preprocess.py


To add new datasets or algorithms, see the instructions in the readme files in those directories.


This code was written using a conda python environment with the following package dependencies

See https://www.google.com/search?q=conda&oq=conda&aqs=chrome..69i57j69i60l2j0l3.889j0j1&sourceid=chrome&ie=UTF-8

cycler                    0.10.0                   py27_0
decorator                 4.0.11                   py27_0
freetype                  2.5.5                         2
functools32               3.2.3.2                  py27_0
icu                       54.1                          0
libpng                    1.6.27                        0
matplotlib                2.0.0               np112py27_0
mkl                       2017.0.1                      0
networkx                  1.11                     py27_0
numpy                     1.12.1                   py27_0
openssl                   1.0.2k                        1
pandas                    0.19.2              np112py27_1
pip                       9.0.1                    py27_1
pyparsing                 2.1.4                    py27_0
pyqt                      5.6.0                    py27_2
python                    2.7.13                        0
python-dateutil           2.6.0                    py27_0
pytz                      2016.10                  py27_0
qt                        5.6.2                         0
readline                  6.2                           2
scikit-learn              0.18.1              np112py27_1
scipy                     0.19.0              np112py27_0
setuptools                27.2.0                   py27_0
sip                       4.18                     py27_0
six                       1.10.0                   py27_0
sqlite                    3.13.0                        0
subprocess32              3.2.7                    py27_0
tk                        8.5.18                        0
wheel                     0.29.0                   py27_0
zlib                      1.2.8                         3
