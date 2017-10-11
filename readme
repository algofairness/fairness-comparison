This repository is meant to facilitate the benchmarking of fairness aware machine learning algorithms.

To run the benchmarks, download the repository, and run the run_metrics.py file.

This will print out metrics for each dataset.


Drawing heavily on original code from:

https://github.com/mbilalzafar/fair-classification

https://github.com/tkamishima/kamfadm



As for a layout of the repository:

  Algorithms each have own dir, with an algorithm class inside, specific to each algorithm (ex: Kamishima, Zafar, etc.)
  Data is found in the data dir, with each data set getting its own sub-dir.
  Audits is where all of the audit results are stored.
  Metrics contains the Metric class that calculates different metrics.
  Misc contains benchmark scripts for Adult, Compas, and German data sets. It also contains necessary files for Zafar, Kamishima, Calders, and Feldman algorithms.
  Preprocessing contains the methods that prepare each data set to be run through the metrics.
  
  To run the working metrics on the Adult and German data sets, cd into the outermost dir and run "python run_metrics.py".


To add a new dataset:
  Dataset must be formatted as csv.
  Make new directory under data and save dataset there.
  Each dataset must have a load_data script which will also be saved in this dir.
  Create preprocessing script for new dataset called prepare_[name] in preprocessing dir.
  Now go into misc/black_box_auditing.py and create classify and repair scripts based on
  the ones present already.
  Finally, go to algorithms/AbstractAlgorithm.py.
  Import prepare script and load_data script.
  Add if statement for [name] with prepare func., name, filename, and classify func.
  Return to outermost directory and edit run_metrics.py
  Add run_metrics([name])

To add new algorithm:
  Create new subdir in algorithms directory.
  Put all files (scripts, etc.) needed to run algorithm into subdirectory.
  Create new Algorithm class in subdir extending AbstractAlgorithm with a run function 
  that runs the algorithm that takes a string representing the name of the dataset (e.g.
  "german") and a dictionary of parameters and returns actual, predicted, and protected 
  lists.
  Import new Algorithm class in run_metrics.py
  Add new section for new algorithm which creates instance of new Algorithm class, runs
  algorithm on data and params, and outputs uniquely named variables for actual,
  predicted, and protected.
  Add new section towards bottom with new instance of Metric calculator that takes
  actual, predicted, and protected vars and runs metrics on them (printed with print_res)











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
