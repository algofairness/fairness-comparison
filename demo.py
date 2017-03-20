import os,sys
import numpy as np
from two_naive_bayes import *
from zafar_classifier import *
from load_dummy_data import *
from prepare_adult_data import *
from prejudice_regularizer import *
sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints



def test_adult_data():
	#Variables for whole functions
	sensitive_attrs = ["sex"]
	train_fold_size = 0.7


	# """ Dummy data to test how my functions are working
	# """
	# print "\n== Dummy data to test classifiers"
	#
	# X, y, x_control = load_dummy_data(20000)
	# X = X.T
	# x_train, y_train, x_control_train, x_test, y_test, x_control_test = split_into_train_test(X, y, x_control, train_fold_size)
	#
	# print "Mutual information in test data: "
	# print_mutual_information(y, x_control, sensitive_attrs)
	# cv_score = compute_cv_score(x_control["sex"], y)
	# print "CV Score test data: %f" % cv_score
	# p_rule = compute_p_rule(x_control["sex"], y)
	# print "P Rule in test data: %f " % p_rule
	#
	# y_classified_results = train_classify(x_train, y_train, x_test, y_test, 1)
	#
	#

	##############################################################################################################################################
	"""
	Load and Split Data
	"""
	##############################################################################################################################################

	""" Load the adult data """
	print "\n"
	# X, y, x_control = load_adult_data("data/adult.data", load_data_size=16000) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	# #X, y, x_control = load_adult_data_from_kamashima() # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	# X = ut.add_intercept(X) # add intercept to X before applying the linear classifier

	X_repair, y_repair, x_control_repair = load_adult_data("data/repair_new.csv", load_data_size=16000) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	#X, y, x_control = load_adult_data_from_kamashima() # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	X_repair = ut.add_intercept(X_repair) # add intercept to X before applying the linear classifier


	""" Split the data into train and test """

	#x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	repair_x_train, repair_y_train, repair_x_control_train, repair_x_test, repair_y_test, repair_x_control_test = ut.split_into_train_test(X_repair, y_repair, x_control_repair, train_fold_size)

	# ##############################################################################################################################################
	# """
	# Classify using Kamishima
	# """
	# ##############################################################################################################################################
	#
	# x_train_with_sensitive_feature = []
	# for i in range(0, len(x_train)):
	# 	val =  x_control_train["sex"][i]
	# 	feature_array = np.append(x_train[i], val)
	# 	x_train_with_sensitive_feature.append(feature_array)
	# x_train_with_sensitive_feature = np.array(x_train_with_sensitive_feature)
	#
	# x_test_with_sensitive_feature = []
	# for i in range(0, len(x_test)):
	# 	val =  x_control_test["sex"][i]
	# 	feature_array = np.append(x_test[i], val)
	# 	x_test_with_sensitive_feature.append(feature_array)
	# x_test_with_sensitive_feature = np.array(x_test_with_sensitive_feature)
	#
	#
	#
	# print "\n== Kamishima's Prejudice Reducer Regularizer with fairness param of 1"
	#
	# y_classified_results = train_classify(x_train_with_sensitive_feature, y_train, x_test_with_sensitive_feature, y_test, 1, 30, x_control_test)
	# #print_mutual_information(y_classified_results, x_control_test, sensitive_attrs)
	# y_classified_results = np.array(y_classified_results)
	#
	# p_rule = compute_p_rule(x_control_test["sex"], y_classified_results)
	# print "P-rule: %f" % p_rule
	# cv_score = compute_cv_score(x_control_test["sex"], y_classified_results)
	# print "CV-score: %f" % cv_score


	##############################################################################################################################################
	"""
	Classify using Calder's Two Naive Bayes
	"""
	##############################################################################################################################################

	# run_two_naive_bayes("original", x_train, y_train, x_control_train, x_test, y_test, x_control_test)
	# print "\n== Calder's Two Naive Bayes =="

	run_two_naive_bayes("new_repair", repair_x_train, repair_y_train, repair_x_control_train, repair_x_test, repair_y_test, repair_x_control_test)
	print "\n== Calder's Two Naive Bayes with repaired data =="

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
	w_uncons = train_test_classifier("unconstrained.txt", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
	apply_accuracy_constraint = 0
	sep_constraint = 0
	sensitive_attrs_to_cov_thresh = {"sex":0}
	print "\n== Zafar:  Classifier with fairness constraint =="
	w_f_cons = train_test_classifier("perfect_fairness.txt", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

	""" Classify such that we optimize for fairness subject to a certain loss in accuracy """
	apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
	apply_accuracy_constraint = 1 # now, we want to optimize fairness subject to accuracy constraints
	sep_constraint = 0
	gamma = 0.5 # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
	print "\n== Zafar:  Classifier with accuracy constraint =="
	w_a_cons = train_test_classifier("opt_fairness.txt", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

	"""
	Classify such that we optimize for fairness subject to a certain loss in accuracy
	In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.

	"""
	apply_fairness_constraints = 0 # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
	apply_accuracy_constraint = 1 # now, we want to optimize accuracy subject to fairness constraints
	sep_constraint = 1 # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
	gamma = 1000.0
	print "\n== Zafar: Classifier with accuracy constraint (no +ve misclassification) =="
	w_a_cons_fine = train_test_classifier("no_positive_misclassification.txt", x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

	##############################################################################################################################################
	"""
	End Zafar Code
	"""
	##############################################################################################################################################




	# """ Classify the data using Kamishima's Prejudice Reducer Regularizer and their own discretized data"""
	# print "\n== Kamishima's Prejudice Reducer Regularizer with their own discretized data"
	#
	# #Load the Kamishima data
	# D = np.loadtxt("kamishima/00DATA/adultd@1t.bindata")
	# #split data and process missing values
	# y_kamishima_data= np.array(D[:, -1])
	#
	# #Switching 0 with -1, for the sake of other classifiers. Keeping track of original data to train the model w/ Kamishima's code.
	# updated_y_kamishima_data = []
	#
	# for j in y_kamishima_data:
	#     if j == 0:
	#         updated_y_kamishima_data.append(-1.0)
	#     else:
	#         updated_y_kamishima_data.append(1.0)
	#
	# updated_y_kamishima_data = np.array(updated_y_kamishima_data)
	# X_kamishima_data = fill_missing_with_mean(D[:, :-1])
	#
	# #TODO Opperating under the assumption that this is the sensitive feature/sex
	# S = np.atleast_2d(D[:, -(1 + number_non_sensative_features):-1])
	# del D
	# sex = []
	# for s in S:
	# 	sex.append(s[0])
	# sex = np.array(sex)
	# S = {'sex': sex}
	#
	#
	# x_train, y_train, x_control_train, x_test, y_test, x_control_test = split_into_train_test(X_kamishima_data, updated_y_kamishima_data, S, train_fold_size)
	#
	# #TODO SO WHY IS THIS CLASSIFICATION BEING SO BAD
	# y_classified_results = train_classify(x_train, y_train, x_test, y_test, 1)
	# x_control_test["sex"] = np.array(x_control_test["sex"])
	#
	# print_mutual_information(y_classified_results, x_control_test, sensitive_attrs)
	# y_classified_results = np.array(y_classified_results)
	# p_rule = compute_p_rule(x_control_test["sex"], y_classified_results)
	# print "P-rule: %f" % p_rule
	# cv_score = compute_cv_score(x_control_test["sex"], y_classified_results)
	# print "CV-score: %f" % cv_score

	# """ Classify the data using Calder's Two Naive Bayes"""
	# print "\n== Calder's Two Naive Bayes Classifier"
	# women_predicted_class_status, women_expected_class_status, men_predicted_class_status, men_expected_class_status = run_two_naive_bayes(x_train, y_train, x_control_train, x_test, y_test, x_control_test)
	#
	# women_predicted_class_status = np.ndarray.tolist(women_predicted_class_status)
	# men_predicted_class_status = np.ndarray.tolist(men_predicted_class_status)
	# all_class_labels_assigned_test = []
	# people_expected_class_status = []
	# x_control_test = {"sex": [ ]}
	# sensitive_attrs = ["sex"]
	#
	#
	# #This is recreating the format needed for print_mutual_information
	# #Which is a list of class values, and a separate list of gender values
	#
	# for idx, val in enumerate(women_predicted_class_status):
	# 	all_class_labels_assigned_test.append(val)
	# 	x_control_test["sex"].append(0)
	#
	# for idx, val in enumerate(men_predicted_class_status):
	# 	all_class_labels_assigned_test.append(val)
	# 	x_control_test["sex"].append(1)
	#
	# for idx, val in enumerate(women_expected_class_status):
	# 	people_expected_class_status.append(val)
	#
	# for idx, val in enumerate(men_expected_class_status):
	# 	people_expected_class_status.append(val)
	#
	# women_positive = 0
	# women_negative = 0
	# for j in women_predicted_class_status:
	# 	if j == -1.0:
	# 		women_negative+=1
	# 	elif j == 1.0:
	# 		women_positive+=1
	# 	else:
	# 		print "Invalid class status in protected class"
	# percent_women_in_positive_class = (float(women_positive) / float(len(women_predicted_class_status)))
	#
	# men_positive = 0
	# men_negative = 0
	# for j in men_predicted_class_status:
	# 	if j == -1.0:
	# 		men_negative+=1
	# 	elif j == 1.0:
	# 		men_positive+=1
	# 	else:
	# 		print "Invalid class status in unprotected class"
	# percent_men_in_positive_class = (float(men_positive) / float(len(men_predicted_class_status)))
	#
	# p_rule = ((percent_women_in_positive_class / percent_men_in_positive_class))*100
	# cv_score = percent_men_in_positive_class - percent_women_in_positive_class
	#
	# accuracy = get_accuracy_for_calders(people_expected_class_status, all_class_labels_assigned_test)
	# correlation_dict_test = get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
	# print_mutual_information(all_class_labels_assigned_test, x_control_test, sensitive_attrs)
	# print "Percent protected/non-protected in +ve class: %f/%f" % (100*percent_women_in_positive_class, 100*percent_men_in_positive_class)
	# print "P-rule achieved: %f" % p_rule
	# print "CV Score: %f" % cv_score
	# print "No covariance for Calders"
	# print "Accuracy: %f" % accuracy

	return

def main():
	test_adult_data()


if __name__ == '__main__':
	main()
