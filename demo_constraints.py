import os,sys
import numpy as np
from prepare_adult_data import *
from two_naive_bayes import *
sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints



def test_adult_data():
	""" Load the adult data """
	X, y, x_control = load_adult_data(load_data_size=10000) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	ut.compute_p_rule(x_control["sex"], y) # compute the p-rule in the original data
	ut.compute_cv_score(x_control["sex"], y) #compute the cv score in the original data

	""" Split the data into train and test """
	X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
	train_fold_size = 0.7
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	apply_fairness_constraints = None
	apply_accuracy_constraint = None
	sep_constraint = None

	loss_function = lf._logistic_loss
	sensitive_attrs = ["sex"]
	sensitive_attrs_to_cov_thresh = {}
	gamma = None

	def train_test_classifier():

		w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
		#W is learned weight vector for the classifier

		#Calculate the accuracy by comparing against correct classification
		train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)

		#Take the dot product of W and each element in the testing set
		distances_boundary_test = (np.dot(x_test, w)).tolist()

		#Classify class labels based off sign (+/-) of result of dot product
		all_class_labels_assigned_test = np.sign(distances_boundary_test)

		correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
		ut.print_mutual_information(all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])

		return w, p_rule, test_score

	def get_accuracy_for_calders(y, Y_predicted):
		correct = []
		assert(len(y) == len(Y_predicted))
		for i in range(len(y)):
			if y[i] == Y_predicted[i]:
				correct.append(1)
			else:
				correct.append(0)

		accuracy = float(sum(correct)) / float(len(correct))
		return accuracy

	""" Classify the data using Calder's Two Naive Bayes"""
	print "\n== Calder's Two Naive Bayes Classifier"
	women_predicted_class_status, women_expected_class_status, men_predicted_class_status, men_expected_class_status = run_two_naive_bayes(x_train, y_train, x_control_train, x_test, y_test, x_control_test)

	women_predicted_class_status = np.ndarray.tolist(women_predicted_class_status)
	men_predicted_class_status = np.ndarray.tolist(men_predicted_class_status)
	all_class_labels_assigned_test = []
	people_expected_class_status = []
	x_control_test = {"sex": [ ]}
	sensitive_attrs = ["sex"]


	#This is recreating the format needed for print_mutual_information
	#Which is a list of class values, and a separate list of gender values

	for idx, val in enumerate(women_predicted_class_status):
		all_class_labels_assigned_test.append(val)
		x_control_test["sex"].append(0)

	for idx, val in enumerate(men_predicted_class_status):
		all_class_labels_assigned_test.append(val)
		x_control_test["sex"].append(1)

	for idx, val in enumerate(women_expected_class_status):
		people_expected_class_status.append(val)

	for idx, val in enumerate(men_expected_class_status):
		people_expected_class_status.append(val)

	women_positive = 0
	women_negative = 0
	for j in women_predicted_class_status:
		if j == -1.0:
			women_negative+=1
		elif j == 1.0:
			women_positive+=1
		else:
			print "Error something is not right"
	percent_women_in_positive_class = (float(women_positive) / float(len(women_predicted_class_status)))

	men_positive = 0
	men_negative = 0
	for j in men_predicted_class_status:
		if j == -1.0:
			men_negative+=1
		elif j == 1.0:
			men_positive+=1
		else:
			print "Error something is not right"
	percent_men_in_positive_class = (float(men_positive) / float(len(men_predicted_class_status)))

	p_rule = (percent_women_in_positive_class / percent_men_in_positive_class) * 100.0
	cv_score = percent_men_in_positive_class - percent_women_in_positive_class

	accuracy = get_accuracy_for_calders(people_expected_class_status, all_class_labels_assigned_test)
	correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
	ut.print_mutual_information(all_class_labels_assigned_test, x_control_test, sensitive_attrs)
	print "Accuracy: %f" % accuracy
	print "Percent protected/non-protected in +ve class: %f/%f" % (100*percent_women_in_positive_class, 100*percent_men_in_positive_class)
	print "P-rule achieved: %f" % p_rule
	print "Haven't done covariance yet!!"
	print "CV Score: %f" % cv_score


	""" Classify the data while optimizing for accuracy """
	print "\n== Zafar: Unconstrained (original) classifier =="
	# all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
	apply_fairness_constraints = 0
	apply_accuracy_constraint = 0
	sep_constraint = 0
	w_uncons, p_uncons, acc_uncons = train_test_classifier()

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
	apply_accuracy_constraint = 0
	sep_constraint = 0
	sensitive_attrs_to_cov_thresh = {"sex":0}
	print "\n== Zafar:  Classifier with fairness constraint =="
	w_f_cons, p_f_cons, acc_f_cons  = train_test_classifier()



	""" Classify such that we optimize for fairness subject to a certain loss in accuracy """
	apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
	apply_accuracy_constraint = 1 # now, we want to optimize fairness subject to accuracy constraints
	sep_constraint = 0
	gamma = 0.5 # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
	print "\n== Zafar:  Classifier with accuracy constraint =="
	w_a_cons, p_a_cons, acc_a_cons = train_test_classifier()

	"""
	Classify such that we optimize for fairness subject to a certain loss in accuracy
	In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.

	"""
	apply_fairness_constraints = 0 # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
	apply_accuracy_constraint = 1 # now, we want to optimize accuracy subject to fairness constraints
	sep_constraint = 1 # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
	gamma = 1000.0
	print "\n== Zafar: Classifier with accuracy constraint (no +ve misclassification) =="
	w_a_cons_fine, p_a_cons_fine, acc_a_cons_fine  = train_test_classifier()

	return

def main():
	test_adult_data()


if __name__ == '__main__':
	main()
