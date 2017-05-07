from __future__ import division
import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    The code will look for the data file in the present directory, if it is not found, it will download them from GitHub.
"""

def check_data_file(fname):
    files = os.listdir(".") # get the current directory listing

def old_load_compas_data(filename):
    """
    "sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",\
    "juv_other_count", "priors_count", "c_charge_degree",
        "two_year_recid"
    """
    #Race should not be used
    #Neither should Sex
    #Add back in age, charge description
    #Remove non-ordered categorical features
    FEATURES_CLASSIFICATION = ["age_cat", "age", "race", "juv_fel_count",
    "juv_other_count", "juv_misd_count", "priors_count"] #features to be used for classification
    CONT_VARIABLES = ["priors_count", "juv_fel_count", "juv_other_count", "juv_misd_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "is_violent_recid" # the decision variable
    SENSITIVE_ATTRS = ["race"]



    COMPAS_INPUT_FILE = filename
    check_data_file(COMPAS_INPUT_FILE)

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])


    """
    What should I do with rows for individuals with missing data regarding charge date
    Right now just turning nan into 0 so not to get errors down the line
    """

    nan_removed = []

    for j in (data["days_b_screening_arrest"]):
        if np.isnan(j):
            nan_removed.append(0.0)
        else:
            nan_removed.append(j)

    nan_removed = np.array(nan_removed)
    data["days_b_screening_arrest"] = nan_removed



    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)
    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_violent_recid"] != -1)
    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]


    """ Feature normalization and one hot encoding """

    y = data[CLASS_FEATURE]

    """
    In ProPublica data we are estimating if an individual commits an act of recitivism
    This is coded as a 1, and if they do not recitivize, a 0.

    In this sense 1 is "negative" and 0 is "positive"

    However the fairness metrics assume 1 is "positive" and 0 is "negative," so inverting the class labels
    """
    print "\nNumber of people violently recidivating within two years"
    print pd.Series(y).value_counts()
    print "\n"

    swapped_y = []
    for value in y:
        if value == 0:
            swapped_y.append(1)
        elif value == 1:
            swapped_y.append(0)
        else:
            print "Incorrect value in class values"
    y = np.array(swapped_y)

    print "\nNumber of people violently recidivating within two years"
    print pd.Series(y).value_counts()
    print "\n"

    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals


        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()


    # """permute the date randomly"""
    # perm = range(0,X.shape[0])
    # shuffle(perm)
    # X = X[perm]
    # y = y[perm]
    # for k in x_control.keys():
    #     x_control[k] = x_control[k][perm]


    X = ut.add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert(len(feature_names) == X.shape[1])
    print "Features we will be using for classification are:", feature_names, "\n"
    return X, y, x_control, feature_names
