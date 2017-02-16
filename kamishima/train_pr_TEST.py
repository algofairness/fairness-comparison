#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training logistic regression

SYNOPSIS::

    SCRIPT [options]

Description
===========

The last column indicates binary class.

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
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
N_NS : int
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

#==============================================================================
# Public symbols
#==============================================================================

__all__ = []

#==============================================================================
# Constants
#==============================================================================

N_NS = 1

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

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
    ns : int
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
# Main routine
#==============================================================================
#Opt values, for reference
#Namespace(C=1.0, eta=30.0, infile=<open file '00DATA/adultd@0l.bindata',
#mode 'r' at 0x107cd24b0>, itype=3, ltype=4, ns=False, ntry=1,
#outfile=<open file '00MODEL/adultd@method=PR4-reg=1-eta=30.0-ltype=4-itype=3-try=1@0l.model', mode 'w' at 0x1089ef420>,
#python_version='2.7.13', rseed=1234, script_name='train_pr.py', script_version='3.0.0',



def main():
    """ Main routine that exits with status code 0
    """
    # init constants
    ns = 1
    N_NS = 1

    ### pre process
    #00DATA/adultd@0l.bindata
    # read data
    D = np.loadtxt("00DATA/adultd@0l.bindata")

    # split data and process missing values
    y = np.array(D[:, -1])
    X = fill_missing_with_mean(D[:, :-1])

    S = np.atleast_2d(D[:, -(1 + N_NS):-1])

    del D


    ### main process

    """
    What are these parameters we're passing in?
    Last four variables in regression_model_with_prejudice_remover: (eta, C, ltype, itype)

    eta, C, ltype, itype
    -C <REG>, --reg <REG>
        regularization parameter (default 1.0)
        I believe this is just the regularizer to avoid overfitting
    -e <eta>, --eta <eta>
        fairness penalty parameter (default 1.0)
        Toggle this parameter to change fairness versus Accuracy
        higher number = more fair but less accurate!
    -l <LTYPE>, --ltype <LTYPE>
        likehood fitting type (default 4)
    -n <ITYPE>, --itype <ITYPE>
        method to initialize coefficients. 0: by zero, 1: at random following
        normal distribution, 2: learned by standard LR, 3: separately learned by
        standard LR (default 3)
    """



    clr = None
    best_loss = np.inf
    best_trial = 0
    #Can run multiple trials to get better results
    for trial in xrange(1):
        #print("Trial No. " + str(trial + 1))
        regression_model_with_prejudice_remover = train(X, y, ns, 30, 1, 4, 3)
        #print("loss = " + str(regression_model_with_prejudice_remover.f_loss_))
        if regression_model_with_prejudice_remover.f_loss_ < best_loss:
            clr = regression_model_with_prejudice_remover
            best_loss = clr.f_loss_
            best_trial = trial + 1

    final_loss = best_loss
    best_trial = best_trial
    ### Going onto what is done in predict_lr.py

    # prediction and write results
    #p is a two-dimensional array, where every element is contains two probabilities
    #corresponding to the binary classification
    #Presumably
    p = clr.predict_proba(X)

    negative = 0
    positive = 0

    for person in y:
        if person == 0:
            negative +=1
        elif person ==1:
            positive +=1

    print "Percent people in positive class in raw data: %f" % (100.0*float(positive)/float(len(y)))
    print "Percent people in negative class in raw data: %f" % (100.0*float(negative)/float(len(y)))


    # output prediction
    total_people = 0
    people_who_were_accurately_classified = 0
    people_in_positive_class = 0
    people_in_negative_class = 0

    for i in xrange(p.shape[0]):
        #print y[i]
        c = np.argmax(p[i, :])
        if c == 0:
            people_in_negative_class+=1
        elif c == 1:
            people_in_positive_class+=1
        total_people += 1
        people_who_were_accurately_classified += 1 if c == y[i] else 0

    print "Percent people classified in positive class: %f" % (100.0*float(people_in_positive_class)/float(total_people))
    print "Percent people classified in negative class: %f" % (100.0*float(people_in_negative_class)/float(total_people))
    print "Accuracy: %f" % (100.0* float(people_who_were_accurately_classified)/float(total_people))

main()
