#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description
===========

Options
=======
-C <REG>, --reg <REG>
    regularization parameter (default 1.0)
-e <eta>, --eta <eta>
    fairness penalty parameter (default 1.0)
-l <LTYPE>, --ltype <LTYPE>
    likehood fitting type (default 4)
-t <NTRY>, --try <NTRY>
    the number of trials with random restart. if 0, all coefficients are
    initialized by zeros, and a model is trained only once. (default 0)
-n <ITYPE>, --itype <ITYPE>
    method to initialize coefficients. 0: by zero, 1: at random following
    normal distribution, 2: learned by standard LR, 3: separately learned by
    standard LR (default 3)
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)
--version
    show version

Attributes
==========
number_non_sensative_features : int
    the number of non sensitive features
"""

#==============================================================================
# Module metadata variables
#==============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2012/08/26"
__version__ = "3.0.0"
__copyright__ = "Copyright (c) 2011 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"
__docformat__ = "restructuredtext en"

#==============================================================================
# Imports
#==============================================================================

import sys
import argparse
import os
import platform
import commands
import logging
import datetime
import pickle
import numpy as np

# private modeules ------------------------------------------------------------
import site
site.addsitedir('.')

from fadm import __version__ as fadm_version
from sklearn import __version__ as sklearn_version
from fadm.util import fill_missing_with_mean
from fadm.lr.pr import *
site.addsitedir('..')

from prepare_adult_data import *
#==============================================================================
# Constants
#==============================================================================
number_sensative_features = 1
number_non_sensative_features = 1
#==============================================================================
# Functions
#==============================================================================
def train(X, y, ns, eta, C, ltype, itype):
    """ train model
    Parameters
    ----------
    X : ary, shape=(n_samples, n_features)
        features
    y : ary, shape=(n_samples)
        classes
    ns, number_sensative_features : int
        the number of sensitive features
    opt : object
        options

    Returns
    -------
    clr : classifier object
        trained classifier
    """
    if ltype == 4:
            clr = LRwPRType4(eta=eta, C=1)
            print clr
            clr.fit(X, y, number_sensative_features, itype)
    else:
        sys.exit("Illegal likelihood fitting type")

    return clr



### main process

def train_classify(X_train, y_train, X_test, y_test, number_sensative_features, fairness_param, x_control_test):

    clr = None
    best_loss = np.inf
    best_trial = 0
    #Can run multiple trials to get better results

    for trial in xrange(1):

        #Check top of file for parameters of regression_model_with_prejudice_remover
        #If you make fairness parameter too big, there will be no women in negative class in Kamishima's
        #Cannot reproduce with fairness parameter at 30, leading me to believe something is wrong (not true but leaving for memory sake)
        regression_model_with_prejudice_remover = train(X_train, y_train, number_sensative_features, fairness_param, 1, 4, 3)
        if regression_model_with_prejudice_remover.f_loss_ < best_loss:
            clr = regression_model_with_prejudice_remover
            best_loss = clr.f_loss_
            best_trial = trial + 1

    final_loss = best_loss
    best_trial = best_trial

    #Calculate predictions
    #p is a two-dimensional array, where every element is contains two probabilities
    #corresponding to the binary classification presumably
    p = clr.predict_proba(X_test)
    negative = 0
    positive = 0
    total_people = len(y_test)

    for person in y_test:
        if person == -1.0 or person == 0.0:
            negative +=1
        elif person ==1.0:
            positive +=1


    print "Percent people in positive class in raw data: %f" % (100.0*float(positive)/float(total_people))
    print "Percent people in negative class in raw data: %f" % (100.0*float(negative)/float(total_people))

    people_who_were_accurately_classified = 0
    people_in_positive_class = 0
    people_in_negative_class = 0
    y_classified_results = []
    for i in xrange(p.shape[0]):
        c = np.argmax(p[i, :])
        if c == 0:
            y_classified_results.append(0)
            people_in_negative_class+=1

            #Because negative classification is represented as -1 and 0 depending on data encoding
            if (y_test[i] == -1.0 or y_test[i] == 0):
                people_who_were_accurately_classified+=1

        elif c == 1:
            y_classified_results.append(1)
            people_in_positive_class+=1
            if y_test[i] == c:
                people_who_were_accurately_classified+=1

    y_test_updated = []
    for j in y_test:
        if j == 1.0:
            y_test_updated.append(1)
        elif j == -1.0 or j == 0.0:
            y_test_updated.append(0)
        else:
            print j
            print "Invalid class value in y_control_test"

    f = open("RESULTS/kamishima:eta="+str(fairness_param), 'w')
    for i in range(0, len(y_test)):
        line_of_data = ( str(y_test_updated[i]) + " " + str(y_classified_results[i]) + " " + str(x_control_test["sex"][i]))
        f.write(line_of_data)
        f.write("\n")
    f.close()


    print "Percent people classified in positive class: %f" % (100.0*float(people_in_positive_class)/float(total_people))
    print "Percent people classified in negative class: %f" % (100.0*float(people_in_negative_class)/float(total_people))
    print "Accuracy: %f" % (100.0* float(people_who_were_accurately_classified)/float(total_people))


    return y_classified_results
