import os,sys
import numpy as np
from two_naive_bayes import *
from zafar_classifier import *
from prejudice_regularizer import *
from black_box_auditing import *
from sklearn import svm
#sys.path.insert(0, "/data/propublica/")
from load_compas import *

sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints


def test_compas_data():
    #Variables for whole functions
    sensitive_attrs = ["race"]
    sensitive_attr = sensitive_attrs[0]
    train_fold_size = 0.7

    ##############################################################################################################################################
    """
    Repair data if needed
    """
    ##############################################################################################################################################

    run_compas_repair()

    ##############################################################################################################################################
    """
    Load and Split Data
    """
    ##############################################################################################################################################

    """ Load the compas data
    y=0: recitivism
    y=1: no recitivism
    """
    print "\n"

    X, y, x_control, feature_names = load_compas_data("compas-scores-two-years-violent-columns-removed.csv") # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup

    X_repaired_8, y_repaired_8, x_control_repaired_8, feature_names = load_compas_data("repaired-compas-scores-two-years-violent-columns-removed_.8.csv") # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
    X_repaired_9, y_repaired_9, x_control_repaired_9, feature_names = load_compas_data("repaired-compas-scores-two-years-violent-columns-removed_.9.csv") # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
    X_repaired_1, y_repaired_1, x_control_repaired_1, feature_names = load_compas_data("repaired-compas-scores-two-years-violent-columns-removed_1.csv") # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup

    # shuffle the data
    perm = range(0,len(y)) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    X_repaired_8 = X_repaired_8[perm]
    X_repaired_9 = X_repaired_9[perm]
    X_repaired_1 = X_repaired_1[perm]

    y = y[perm]

    for k in x_control.keys():
        x_control[k] = x_control[k][perm]


    """ Split the data into train and test """
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

    x_train_8, y_train_8, x_control_train_8, x_test_8, y_test_8, x_control_test_8 = ut.split_into_train_test(X_repaired_8, y_repaired_8, x_control_repaired_8, train_fold_size)
    x_train_9, y_train_9, x_control_train_9, x_test_9, y_test_9, x_control_test_9 = ut.split_into_train_test(X_repaired_9, y_repaired_9, x_control_repaired_9, train_fold_size)
    x_train_1, y_train_1, x_control_train_1, x_test_1, y_test_1, x_control_test_1 = ut.split_into_train_test(X_repaired_1, y_repaired_1, x_control_repaired_1, train_fold_size)


    ##############################################################################################################################################
    """
    Naive Bayes, Logistic Regression, and SVM on Original/Repaired Data
    """
    ##############################################################################################################################################

    classify_compas("compas_repaired_.9", sensitive_attr, x_train_8, y_train, x_control_train, x_test_9, y_test, x_control_test)
    classify_compas("compas_repaired_.8", sensitive_attr, x_train_9, y_train, x_control_train, x_test_9, y_test, x_control_test)
    classify_compas("compas_repaired_1", sensitive_attr, x_train_1, y_train, x_control_train, x_test_9, y_test, x_control_test)

    classify_compas("compas_original", sensitive_attr, x_train, y_train, x_control_train, x_test, y_test, x_control_test)

    print "SVM, NB, LR on Repaired/Original Data"



    ##############################################################################################################################################
    """
    Classify using Calder's Two Naive Bayes
    """
    ##############################################################################################################################################

    run_two_naive_bayes("pro_publica_two_naive_bayes", x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attr)
    print "\n== Calder's Two Naive Bayes =="

    ##############################################################################################################################################
    """
    Classify using Kamishima
    """
    ##############################################################################################################################################

    x_train_with_sensitive_feature = []
    for i in range(0, len(x_train)):
        val =  x_control_train[sensitive_attr][i]
        feature_array = np.append(x_train[i], val)
        x_train_with_sensitive_feature.append(feature_array)
    x_train_with_sensitive_feature = np.array(x_train_with_sensitive_feature)

    x_test_with_sensitive_feature = []
    for i in range(0, len(x_test)):
        val =  x_control_test[sensitive_attr][i]
        feature_array = np.append(x_test[i], val)
        x_test_with_sensitive_feature.append(feature_array)
    x_test_with_sensitive_feature = np.array(x_test_with_sensitive_feature)


    print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 30"

    y_classified_results = train_classify(sensitive_attr, "compas", x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 30, x_control_test)

    print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 15"

    y_classified_results = train_classify(sensitive_attr, "compas", x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 15, x_control_test)

    print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 1"

    y_classified_results = train_classify(sensitive_attr, "compas", x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 1, x_control_test)


    ##############################################################################################################################################
    """
    Zafar Code
    """
    ##############################################################################################################################################
    #Variables for Zafar classifiers
    apply_fairness_constraints = None
    apply_accuracy_constraint = None
    sep_constraint = None
    loss_function = lf._logistic_loss
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    """ Classify the data while optimizing for accuracy """
    print "\n== Zafar: Unconstrained (original) classifier =="
    # all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0
    w_uncons = train_test_classifier("pro_publica_unconstrained", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    """ Now classify such that we optimize for accuracy while achieving perfect fairness """
    apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
    apply_accuracy_constraint = 0
    sep_constraint = 0
    sensitive_attrs_to_cov_thresh = {"race":0}
    print "\n== Zafar:  Classifier with fairness constraint =="
    w_f_cons = train_test_classifier("pro_publica_opt_accuracy", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    """ Classify such that we optimize for fairness subject to a certain loss in accuracy """
    apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
    apply_accuracy_constraint = 1 # now, we want to optimize fairness subject to accuracy constraints
    sep_constraint = 0
    gamma = 0.5 # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
    print "\n== Zafar:  Classifier with accuracy constraint =="
    w_a_cons = train_test_classifier("pro_publica_opt_fairness", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    """
    Classify such that we optimize for fairness subject to a certain loss in accuracy
    In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.

    """
    apply_fairness_constraints = 0 # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
    apply_accuracy_constraint = 1 # now, we want to optimize accuracy subject to fairness constraints
    sep_constraint = 1 # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
    gamma = 1000.0
    print "\n== Zafar: Classifier with accuracy constraint (no +ve misclassification) =="
    w_a_cons_fine = train_test_classifier("pro_publica_no_positive_misclassification", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    ##############################################################################################################################################
    """
    End Zafar Code
    """
    ##############################################################################################################################################


    return

def main():
    test_compas_data()


if __name__ == '__main__':
    main()
