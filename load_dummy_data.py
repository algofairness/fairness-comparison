import numpy as np
from sklearn.utils import shuffle

def load_dummy_data(data_size):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['att1', 'att2', 'att3', 'sex'] # all attributes
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['sex'] #  will not use them in classification
    attrs_for_classification = set(attrs) #- set(attrs_to_ignore)



    X = []
    y = []
    x_control = {"sex": []}


    for f in range(data_size/2):
        if f % 2 == 0:
            a = [1, 1, 1]
            a = np.array(a)
            X.append(a)
            x_control["sex"].append(1.0)
            y.append(1)
        else:
            a = [1, 1, 1]
            a = np.array(a)
            X.append(a)
            x_control["sex"].append(0.0)
            y.append(-1)

    for f in range(data_size/2):
        if f % 2 == 0:
            a = [0, 0, 0]
            a = np.array(a)
            X.append(a)
            x_control["sex"].append(1.0)
            y.append(1)
        else:
            a = [0, 0, 0]
            a = np.array(a)
            X.append(a)
            x_control["sex"].append(0.0)
            y.append(-1)

    for f in range(50):
        a = [1, 0, 1]
        a = np.array(a)
        X.append(a)
        x_control["sex"].append(0.0)
        y.append(1)
    for f in range(50):
        a = [1, 0, 1]
        a = np.array(a)
        X.append(a)
        x_control["sex"].append(1.0)
        y.append(-1)


    #Shuffling data so it doesn't break scipy
    x_control_shuffled = {"sex": []}
    X_shuffled, y_shuffled, x_control_shuffled["sex"] = shuffle(X, y, x_control["sex"])

    # convert to numpy arrays for easy handling
    X = np.array(X_shuffled, dtype=float).T
    y = np.array(y_shuffled, dtype = float)
    for k, v in x_control_shuffled.items(): x_control_shuffled[k] = np.array(v, dtype=float)


    return X, y, x_control_shuffled
