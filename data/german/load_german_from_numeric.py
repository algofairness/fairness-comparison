import numpy as np
from sklearn import preprocessing

"""
0 = women
1 = man
1 = good credit
0 = bad credit
"""

def load_german_data(filename):
    X = []
    y = []
    x_control = []
    x_vals_to_convert = {}
    headers = "One,Two,three,four,Five,Six,Seven,eight,nine,Ten,Eleven,Twelve,13,14,15,16,17,18,19,20,21,22,23"
    headers = headers.split(",")
    print headers
    for k in headers:
        x_vals_to_convert[k] = []
    for line in open("data/german/"+filename):
        line = line.strip()
        if line == "": continue # skip empty lines

        #This should be more programatic
        if line[0] == "O": continue # skip line of feature categories, in csv

        line = line.split(",")

        """
        Get class label
        """
        class_label = line[-1]
        y.append(class_label)


        """
        Get sex/sensitive variable
        """

        gender = line[-2]
        x_control.append(gender)

        """
        Set rest of variables as X
        """
        for i in range(0, len(line)-2):
            x_vals_to_convert[headers[i]].append(line[i])
        # x_val = line[0:-2]
        #X.append(x_val)

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

    for k in range(10):
        print X[k]
        print len(X[k])
        print "\n"

    x_control = {"sex": x_control}
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    y = y.astype(float)


    return X, y, x_control
