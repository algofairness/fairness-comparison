import os,sys
import urllib2
sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut
import numpy as np
from fadm.util import *
from fairness_metric_calculations import *
from random import seed, shuffle
from sklearn import preprocessing
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: http://archive.ics.uci.edu/ml/datasets/Adult
    The code will look for the data files (adult.data, adult.test) in the present directory, if they are not found, it will download them from UCI archive.
"""

def load_adult_data(filename, load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    #attrs = age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income-per-year
    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', "native-country", "income-per-year"]# "income-per-year"] # all attributes
    int_attrs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['fnlwgt', 'sex', 'education', 'marital-status', 'occupation', 'relationship', 'race', "workclass", "native-country"]
    #attrs_to_ignore = ['sex', 'race' ,'fnlwgt'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = [filename]



    X = []
    y = []                          #Class label: -1 is <=50K, +1 is >50K
    x_control = {}                  #X_control: Sex value of individuals
    attribute_type = dict()         #Dictionary of possible attribute types

    for att in attrs:
        attribute_type[att] = []   #Initialize attribute types

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:

        for line in open(f):
            line = line.strip()
            if line == "": continue # skip empty lines
            if line[0] == "a": continue # skip line of feature categories, in csv

            line = line.split(",")


            class_label = line[-1]
            if class_label == ">50K":
                y.append(1)
            else:
                y.append(0)

            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]

                #reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"

                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)



    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():

            #Scale int_attrs
            if attr_name in int_attrs:
                print attr_name
                print attr_vals[10:50]

                #Numpy and Sklearn gymnastics to scale values
                attr_vals=np.array(attr_vals)
                attr_vals = attr_vals.reshape(-1, 1)

                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
                min_max_scaled = min_max_scaler.fit_transform(attr_vals)
                min_max_scaler.fit(attr_vals)
                scaled_attr_val = min_max_scaler.transform(attr_vals)
                scaled_attr_val = scaled_attr_val.ravel()
                scaled_attr_val = scaled_attr_val.tolist()

                #Getting back to 1D python list
                d[attr_name] = scaled_attr_val
                #d[attr_name] = attr_vals

            else:

                uniq_vals = sorted(list(set(attr_vals))) # get unique values

                # compute integer codes for the unique values
                val_dict = {}
                for i in range(0,len(uniq_vals)):
                    val_dict[uniq_vals[i]] = i

                # replace the values with their integer encoding
                for i in range(0,len(attr_vals)):
                    attr_vals[i] = val_dict[attr_vals[i]]

                #if attr_name != "sex":
                d[attr_name] = attr_vals


    # convert the discrete values to their integer representations
    #Not part of convert_attrs_to_ints function


    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    #One-hot encoding takes categorical data and encodes it as yes/no binary membership, creating many additional columns
    # if the integer vals are categorical and not binary, we need to get one-hot encoding for them

    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else:
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)

    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    # see if we need to subsample the data
    if load_data_size is not None:
        print "Loading only %d examples from the adult dataset" % load_data_size
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]


    return X, y, x_control
