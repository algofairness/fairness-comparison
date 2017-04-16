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

def load_compas_data(filename):
    X = []
    y = []
    x_control = []
    x_vals_to_convert = {}
    headers = "sex,age,age_cat,juv_fel_count,juv_misd_count,juv_other_count,priors_count,c_charge_degree"
    headers = headers.split(",")

    print headers

    for k in headers:
        x_vals_to_convert[k] = []
    for line in open("data/propublica/"+filename):
        line = line.strip()
        if line == "": continue # skip empty lines

        #This should be more programatic
        if line[0] == "s": continue # skip line of feature categories, in csv

        line = line.split(",")

        """
        Get class label
        """
        class_label = line[-2]
        y.append(class_label)


        """
        Get sex/sensitive variable
        """

        race = line[-1]
        x_control.append(race)

        """
        Set rest of variables as X
        """
        for i in range(0, len(line)-2):
            x_vals_to_convert[headers[i]].append(line[i])

    for attr_name, attr_vals in x_vals_to_convert.items():

        attr_vals=np.array(attr_vals)
        attr_vals = attr_vals.reshape(-1, 1)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        min_max_scaled = min_max_scaler.fit_transform(attr_vals)
        min_max_scaler.fit(attr_vals)
        scaled_attr_val = min_max_scaler.transform(attr_vals)
        scaled_attr_val = scaled_attr_val.ravel()
        scaled_attr_val = scaled_attr_val.tolist()
        #Getting back to 1D python list

        X.append(scaled_attr_val)

        #X.append(attr_vals)

    X = np.array(X, dtype=float).T

    """
    In ProPublica data we are estimating if an individual commits an act of recitivism
    This is coded as a 1, and if they do not recitivize, a 0.

    In this sense 1 is "negative" and 0 is "positive"

    However the fairness metrics assume 1 is "positive" and 0 is "negative," so inverting the class labels
    """

    swapped_y = []
    for value in y:
        if int(value) == 0:
            swapped_y.append(1)
        elif int(value) == 1:
            swapped_y.append(0)
        else:
            print "Incorrect value in class values"
    y = np.array(swapped_y)

    x_control = {"race": x_control}
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    y = y.astype(float)


    return X, y, x_control
