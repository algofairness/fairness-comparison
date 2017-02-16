#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description
===========

The last column indicates binary class.

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
    number_sensative_features : int
        the number of sensitive features
    opt : object
        options

    Returns
    -------
    clr : classifier object
        trained classifier
    """
    if ltype == 4:
            #clr = LRwPRType4(eta=opt.eta, C=opt.C)
            clr = LRwPRType4(eta, C)
            clr.fit(X, y, ns, itype)
            #clr.fit(X, y, ns, itype=opt.itype)
    else:
        sys.exit("Illegal likelihood fitting type")

    return clr


#==============================================================================
# Load Data
#==============================================================================
D = np.loadtxt("kamishima/00DATA/adultd@1t.bindata")
#split data and process missing values
y_kamishima_data= np.array(D[:, -1])
updated_y_kamishima_data = []

for j in y_kamishima_data:
    if j == 0:
        updated_y_kamishima_data.append(-1.0)
    else:
        updated_y_kamishima_data.append(1)

X_kamishima_data = fill_missing_with_mean(D[:, :-1])
#TODO WHAT IS THIS: Is it sex?
S = np.atleast_2d(D[:, -(1 + number_non_sensative_features):-1])
del D


### main process

def train_classify(X, y, number_sensative_features):
    clr = None
    best_loss = np.inf
    best_trial = 0
    #Can run multiple trials to get better results
    for trial in xrange(1):

        #Check top of file for parameters of regression_model_with_prejudice_remover
        regression_model_with_prejudice_remover = train(X, y, number_sensative_features, 30, 1, 4, 3)
        if regression_model_with_prejudice_remover.f_loss_ < best_loss:
            clr = regression_model_with_prejudice_remover
            best_loss = clr.f_loss_
            best_trial = trial + 1

    final_loss = best_loss
    best_trial = best_trial

    #Calculate predictions
    #p is a two-dimensional array, where every element is contains two probabilities
    #corresponding to the binary classification presumably
    p = clr.predict_proba(X)

    negative = 0
    positive = 0

    for person in y:
        if person == -1.0:
            negative +=1
        elif person ==1.0:
            positive +=1

    print "Percent people in positive class in raw data: %f" % (100.0*float(positive)/float(len(y)))
    print "Percent people in negative class in raw data: %f" % (100.0*float(negative)/float(len(y)))

    total_people = 0
    people_who_were_accurately_classified = 0
    people_in_positive_class = 0
    people_in_negative_class = 0
    total_people = len(y)

    for i in xrange(p.shape[0]):
        c = np.argmax(p[i, :])
        #print (" ".join(S[i, :].astype(str)) + " ")

        if c == 0:
            people_in_negative_class+=1
            #Because negative classification is represented as -1 and 0 depending on data encoding
            if (y[i] == -1.0 or y[i] == 0):
                people_who_were_accurately_classified+=1
        elif c == 1:
            people_in_positive_class+=1
            if y[i] == c:
                people_who_were_accurately_classified+=1

    print "Percent people classified in positive class: %f" % (100.0*float(people_in_positive_class)/float(total_people))
    print "Percent people classified in negative class: %f" % (100.0*float(people_in_negative_class)/float(total_people))
    print "Accuracy: %f" % (100.0* float(people_who_were_accurately_classified)/float(total_people))

#print "\nWith data discretized by Zafar code: \n"
#train_classify(X, y, 1)
#print "\nWith Kamishima's discretization: \n"
#train_classify(X_kamishima_data, y_kamishima_data, 1)
