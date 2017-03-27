import os,sys
import numpy as np
from two_naive_bayes import *
from zafar_classifier import *
from load_dummy_data import *
from prepare_adult_data import *
from prejudice_regularizer import *
from black_box_auditing import *
from load_compas_data import *
from sklearn import svm

sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints




def test_adult_data():
    #Variables for whole functions
    sensitive_attrs = ["race"]
    train_fold_size = 0.5


    ##############################################################################################################################################
    """
    Load and Split Data
    """
    ##############################################################################################################################################

    """ Load the adult data """
    print "\n"
    X, y, x_control = load_compas_data() # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
    sensitive_attrs = x_control.keys()
    sensitive_attr = sensitive_attrs[0]

    """ Split the data into train and test """

    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

    # ##############################################################################################################################################
    # """
    # Write/re-encode the audit data
    # """
    # ##############################################################################################################################################
    #
    # encode_blackbox_audit()

    # ##############################################################################################################################################
    # """
    # Classify using Kamishima
    # """
    # ##############################################################################################################################################
    #
    # x_train_with_sensitive_feature = []
    # for i in range(0, len(x_train)):
    #     val =  x_control_train["sex"][i]
    #     feature_array = np.append(x_train[i], val)
    #     x_train_with_sensitive_feature.append(feature_array)
    # x_train_with_sensitive_feature = np.array(x_train_with_sensitive_feature)
    #
    # x_test_with_sensitive_feature = []
    # for i in range(0, len(x_test)):
    #     val =  x_control_test["sex"][i]
    #     feature_array = np.append(x_test[i], val)
    #     x_test_with_sensitive_feature.append(feature_array)
    # x_test_with_sensitive_feature = np.array(x_test_with_sensitive_feature)
    #
    #
    # print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 30"
    #
    # y_classified_results = train_classify(x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 30, x_control_test)

    # print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 15"
    #
    # y_classified_results = train_classify(x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 15, x_control_test)
    #
    # print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 1"
    #
    # y_classified_results = train_classify(x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 1, x_control_test)
    #

    ##############################################################################################################################################
    """
    Classify using standard SVM
    """
    ##############################################################################################################################################

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    """
    In ProPublica data we are estimating if an individual commits an act of recitivism
    This is coded as a 1, and if they do not recitivize, a 0.

    In this sense 1 is "negative" and 0 is "positive"

    However the fairness metrics assume 1 is "positive" and 0 is "negative," so inverting the class labels
    """
    converted_y_test = []
    for value in y_test:
        if value == 0:
            converted_y_test.append(1)
        elif value == 1:
            converted_y_test.append(0)
        else:
            print "Incorrect value in class values"

    converted_assigned_class_values = []
    for value in predictions:
        if value == 1:
            converted_assigned_class_values.append(0)
        elif value == 0:
            converted_assigned_class_values.append(1)
        else:
            print "Incorrect value in class values"


    f = open("results/pro_publica_svm_new", 'w')
    for i in range(0, len(converted_y_test)-1):
        line_of_data = (str (converted_y_test[i]) + " " + str(converted_assigned_class_values[i]) + " " + str(float(x_control_test[sensitive_attr][i])))
        f.write(line_of_data)
        f.write("\n")
    f.close()


    ##############################################################################################################################################
    """
    Classify using repaired data and standard SVM
    """
    ##############################################################################################################################################

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    """
    In ProPublica data we are estimating if an individual commits an act of recitivism
    This is coded as a 1, and if they do not recitivize, a 0.

    In this sense 1 is "negative" and 0 is "positive"

    However the fairness metrics assume 1 is "positive" and 0 is "negative," so inverting the class labels
    """
    converted_y_test = []
    for value in y_test:
        if value == 0:
            converted_y_test.append(1)
        elif value == 1:
            converted_y_test.append(0)
        else:
            print "Incorrect value in class values"

    converted_assigned_class_values = []
    for value in predictions:
        if value == 1:
            converted_assigned_class_values.append(0)
        elif value == 0:
            converted_assigned_class_values.append(1)
        else:
            print "Incorrect value in class values"


    f = open("results/pro_publica_svm_new", 'w')
    for i in range(0, len(converted_y_test)-1):
        line_of_data = (str (converted_y_test[i]) + " " + str(converted_assigned_class_values[i]) + " " + str(float(x_control_test[sensitive_attr][i])))
        f.write(line_of_data)
        f.write("\n")
    f.close()


    ##############################################################################################################################################
    """
    Classify using Calder's Two Naive Bayes
    """
    ##############################################################################################################################################

    run_two_naive_bayes("pro_publica_two_naive_bayes", x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attr)
    print "\n== Calder's Two Naive Bayes =="

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
    sensitive_attrs_to_cov_thresh = {"sex":0}
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
    test_adult_data()


if __name__ == '__main__':
    main()
