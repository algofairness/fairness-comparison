import os,sys
import numpy as np
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import json

def train_classifier(x, y, control, sensitive_attrs, mode, sensitive_attrs_to_cov_thresh):
    loss_function = lf._logistic_loss
    w = ut.train_model(
        x, y, control, loss_function,
        mode.get('fairness', 0),
        mode.get('accuracy', 0),
        mode.get('separation', 0),
        sensitive_attrs,
        sensitive_attrs_to_cov_thresh,
        mode.get('gamma', None))
    return w

def get_accuracy(y, Y_predicted):
    correct_answers = (Y_predicted == y).astype(int) # will have 1 when the prediction and the actual label match
    accuracy = float(sum(correct_answers)) / float(len(correct_answers))
    return accuracy, sum(correct_answers)

def predict(model, x):
    return np.sign(np.dot(x, model))

def check_accuracy(model, x, y):
    predicted = np.sign(np.dot(x, model))
    return get_accuracy(y, predicted)

def test_classifier(w, x, y, control, sensitive_attrs):
    distances_boundary_test = (np.dot(x, w)).tolist()
    all_class_labels_assigned_test = np.sign(distances_boundary_test)
    test_score, _ = check_accuracy(w, x, y)
    correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, control, sensitive_attrs)
    cov_dict_test = ut.print_covariance_sensitive_attrs(None, x, distances_boundary_test, control, sensitive_attrs)
    p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])
    return p_rule, test_score

def load_json(filename):
    f = json.load(open(filename))
    x = np.array(f["x"])
    y = np.array(f["class"])
    sensitive = dict((k, np.array(v)) for (k,v) in f["sensitive"].items())
    return x, y, sensitive

def main(train_file, test_file, output_file, setting, value):
    x_train, y_train, x_control_train = load_json(train_file)
    x_test, y_test, x_control_test = load_json(test_file)

    # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    x_train = ut.add_intercept(x_train)
    x_test = ut.add_intercept(x_test)

    # x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, 0.7)

    # print >> sys.stderr, "First row:"
    # print >> sys.stderr, x_train[0,:], y_train[0], x_control_train

    if setting == 'gamma':
        mode = {"accuracy": 1, "gamma": float(value)}
    elif setting == 'c':
        mode = {"fairness": 1}
    elif setting == 'baseline':
        mode = {}
    else:
        raise Exception("Don't know how to handle setting %s" % setting)

    thresh = {}
    if setting == 'c':
        thresh = dict((k, float(value)) for (k, v) in x_control_train.items())
        # print("Covariance threshold: %s" % thresh)

    # print("Will train classifier on %s %s-d points" % x_train.shape, file=sys.stderr)
    # print("Sensitive attribute: %s" % (x_control_train.keys(),), file=sys.stderr)
    sensitive_attrs = list(x_control_train.keys())
    w = train_classifier(x_train, y_train, x_control_train,
                         sensitive_attrs, mode,
                         thresh)
                         
    # print("Model trained successfully.", file=sys.stderr)

    predictions = predict(w, x_test).tolist()
    output_file = open(output_file, "w")
    json.dump(predictions, output_file)
    output_file.close()

    

##############################################################################
# we prefer simple IO to efficient IO, so everything goes in json

# from prepare_adult_data import *
def write_adult_data_to_disk():
    X, y, x_control = load_adult_data() # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, 0.7)

    adult_train = {"x": x_train.tolist(),
                   "class": y_train.tolist(),
                   "sensitive": {"sex": x_control_train["sex"].tolist()}}
    adult_test = {"x": x_test.tolist(),
                   "class": y_test.tolist(),
                   "sensitive": {"sex": x_control_test["sex"].tolist()}}
    train_out = open("adult_train.json", "w")
    json.dump(adult_train, train_out)
    train_out.close()
    test_out = open("adult_test.json", "w")
    json.dump(adult_test, test_out)
    test_out.close()

if __name__ == '__main__':
    main(*sys.argv[1:])
    exit(0)
