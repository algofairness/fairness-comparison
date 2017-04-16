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

def load_adult_data(filename, load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'] # all attributes
    int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['fnlwgt', 'sex', 'race']
    #attrs_to_ignore = ['fnlwgt', 'sex', 'education', 'marital_status', 'occupation', 'relationship', 'race', "workclass", "native_country"]
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = filename



    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    print "Hey Evan"


    for f in data_files:
        # check_data_file(f)

        for line in open(f):
            line = line.strip()
            if line == "": continue # skip empty lines
            line = line.split(",")
            if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
                continue
            if len(line) != 15 or "a" in line[0]: # if a line has missing attributes, ignore it
                continue

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = -1
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)


            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
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

            if attr_name in int_attrs:

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
            else:
                uniq_vals = sorted(list(set(attr_vals))) # get unique values

                # compute integer codes for the unique values
                val_dict = {}
                for i in range(0,len(uniq_vals)):
                    val_dict[uniq_vals[i]] = i

                # replace the values with their integer encoding
                for i in range(0,len(attr_vals)):
                    attr_vals[i] = val_dict[attr_vals[i]]
                d[attr_name] = attr_vals


    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)


    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else:
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)


    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    # shuffle the data
    perm = range(0,len(y)) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print "Loading only %d examples from the data" % load_data_size
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control
