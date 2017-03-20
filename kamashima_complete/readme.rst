Fairness-aware Data Classification
==================================

:Author: `Toshihiro Kamishima <http://www.kamishima.net/>`_
:Copyright: Copyright (c) 2012 Toshihiro Kamishima all rights reserved.
:License: `MIT License <http://www.opensource.org/licenses/mit-license.php>`_
:Homepage: http://www.kamishima.net/fadm/

This is a test code for an fairness-aware classification .   This code provides the results of Table 1 and Figure 1 in [ECMLPKDD2012]_.   By following the instruction, you will be able to obtain evaluation statistics.   We would like you to cite some of our publications if you utilize these scripts.

.. [ECMLPKDD2012] T. Kamishima, S. Akaho, H. Asoh, and J. Sakuma "Fairness-Aware Classifier with Prejudice Remover Regularizer" Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD), Part II, pp.35-50 (2012)

Instruction
-----------

- We tested this script under the following environment:
  - Python 2.7.x
  - NumPy 1.6.x
  - SciPy 0.10.x

- Generate the "adultd" data sets, which is distributed at the same page that this script is distributed.

- Copy ``adultd.data`` and ``adultd.bindata`` to the ``00DATA`` directory.

- generate datasets for each fold in cross validations. To do this, run the command at the ``00DATA`` directory::

    python generate_cvtest.py adultd.data -o adultd -e data -f 5
    python generate_cvtest.py adultd.bindata -o adultd -e bindata -f 5

- Execute sh script, ``test-all-task.sh``.   A job list, ``gxp-all-task``, is available for the users of the `gxp <http://www.logos.ic.i.u-tokyo.ac.jp/gxp/>`_ shell, which is for running scripts on cluster machines.   Because these scripts are very slow, you'd better to run on multiple CPUs.

Finally, by running script, ``summary-result.sh``, you will obtain file ``adultd@t.txt`` containing evaluation indexes in the ``00SUMMARY`` directly, which already contains our computation results ``adultd@t.txt.orig``.

evaluation statistics file format
---------------------------------

The first column in the file ``adultd@t.txt`` show the experimental condititon.   A keyword ``method`` express the method as follows

* CV2NB : Calders and Verwer's 2-naive-Bayes
* LR : logistic regression with a sensitive feature
* LRns : logistic regression without a sensitive feature
* NB : naive Bayes with a sensitive feature
* NBns : naive Bayes without a sensitive feature
* PR4 : logistic regression with prejudice remover regularizer

The other keywords denotes the statuses of parameters.   The meanings of the other columns are as follows.

=== ==== =====================================================================
col stat description
=== ==== =====================================================================
1   cond experimental condition
11  Acc  cccuracy
21  MI   mutual information between an estimated class and a sample class
25  NMI  normalized version of the column 22
47  PI   mutual information between an estimated class and a sensitive feature
51  NPI  normalized version of the column 47
55  UEI  UEI
56  SCVS Calders-Verwer score in terms of a sample class
57  ECVS Calders-Verwer score in terms of a estimated class
=== ==== =====================================================================
