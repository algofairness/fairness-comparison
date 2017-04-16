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
from fadm.nb.cv2nb import *
site.addsitedir('..')

from prepare_adult_data import *
#==============================================================================
# Constants
#==============================================================================
N_CLASSES = 2
N_S_VALUES = 2

def train_nb_classify(sensitive_attr, dataname, X_train, y_train, X_test, y_test, x_control_test):

    nb = CaldersVerwerTwoNaiveBayes(13, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    nb.fit(X_train, y_train)


    #Calculate predictions
    #p is a two-dimensional array, where every element is contains two probabilities
    #corresponding to the binary classification presumably
    p = nb._predict_log_proba_upto_const(X_test)
    print p


    # print "Percent people in positive class in raw data: %f" % (100.0*float(positive)/float(total_people))
    # print "Percent people in negative class in raw data: %f" % (100.0*float(negative)/float(total_people))
    #
    # people_who_were_accurately_classified = 0
    # people_in_positive_class = 0
    # people_in_negative_class = 0
    # y_classified_results = []
    # for i in xrange(p.shape[0]):
    #     c = np.argmax(p[i, :])
    #     if c == 0:
    #         y_classified_results.append(0)
    #         people_in_negative_class+=1
    #
    #         #Because negative classification is represented as -1 and 0 depending on data encoding
    #         if (y_test[i] == -1.0 or y_test[i] == 0):
    #             people_who_were_accurately_classified+=1
    #
    #     elif c == 1:
    #         y_classified_results.append(1)
    #         people_in_positive_class+=1
    #         if y_test[i] == c:
    #             people_who_were_accurately_classified+=1
    #
    # y_test_updated = []
    # for j in y_test:
    #     if j == 1.0:
    #         y_test_updated.append(1)
    #     elif j == -1.0 or j == 0.0:
    #         y_test_updated.append(0)
    #     else:
    #         print j
    #         print "Invalid class value in y_control_test"
    #
    # f = open("RESULTS/"+str(dataname)+ "kamishima:eta="+str(fairness_param), 'w')
    # for i in range(0, len(y_test)):
    #     line_of_data = ( str(y_test_updated[i]) + " " + str(y_classified_results[i]) + " " + str(x_control_test[sensitive_attr][i]))
    #     f.write(line_of_data)
    #     f.write("\n")
    # f.close()
    #
    #
    # print "Percent people classified in positive class: %f" % (100.0*float(people_in_positive_class)/float(total_people))
    # print "Percent people classified in negative class: %f" % (100.0*float(people_in_negative_class)/float(total_people))
    # print "Accuracy: %f" % (100.0* float(people_who_were_accurately_classified)/float(total_people))
    #
    #
    # return y_classified_results
