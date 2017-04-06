from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
import urllib2


def splitDataByGender(X, x_control, y):

    """
    X is a list of individuals, x_control is a list of each individual's sex (as binary variable),
    y is a list of that individual's actual class status as binary variable (where 0 = income <=50k, 1 = income >50k)
    X[i], x_control[i], y[i] should all correspond to the same individual. Accordingly each list should be of same length
    """

    #Converting from numpy list to standard python list
    X.tolist()
    #x_control.tolist()
    y.tolist()
    women = []
    y_women = []
    men = []
    y_men = []
    #Loop through for every individual, split into two lists by gender
    for i in range(len(X)):

        #Converting from numpy list to standard python list
        person = X[i].tolist()
        person.append(x_control[i])

        if (x_control[i] == 0.0):
            women.append(person)
            y_women.append(y[i])
        else:
            men.append(person)
            y_men.append(y[i])

    return women, y_women, men, y_men

def splitDataBySensitiveFeature(X, x_control, y, sensitive_attr):

    """
    X is a list of individuals, x_control is a list of each individual's sensitive variable (as binary variable),
    y is a list of that individual's actual class status as binary variable (where 0 = income <=50k, 1 = income >50k)
    X[i], x_control[i], y[i] should all correspond to the same individual. Accordingly each list should be of same length
    """

    #Converting from numpy list to standard python list
    X = np.array(X, dtype=float)

    negative = []
    y_negative = []
    positive = []
    y_positive = []

    #Loop through for every individual, split into two lists by gender
    for i in range(len(X)):

        #Converting from numpy list to standard python list
        person = X[i].tolist()
        person.append(x_control[i])
        ##What should this be

        if (float(x_control[i]) == 0.0):
            negative.append(person)
            y_negative.append(float(y[i]))
        else:
            positive.append(person)
            y_positive.append(float(y[i]))

    return negative, y_negative, positive, y_positive



def predict(X, y, X_test, y_test):

    # fit a Naive Bayes model to the data
    model = GaussianNB()
    X = np.array(X).astype(float)
    y = np.array(y).astype(float)
    X_test = np.array(X_test).astype(float)
    y_test = np.array(y_test).astype(float)

    model.fit(X, y)

    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    predicted = predicted.tolist()

    #Replacing -1.0 with 0, for sake of compatability with Kamashima's code

    updated_predicted = []
    for i in predicted:
        if i == -1.0:
            updated_predicted.append(0)
        elif i == 1.0:
            updated_predicted.append(1)
        elif i == 0.0:
            updated_predicted.append(0)

        else:
            print "Inproper value in predicted class values"

    updated_expected = []
    for i in expected:
        if i == -1.0:
            updated_expected.append(0)
        elif i == 1.0:
            updated_expected.append(1)
        elif i == 0.0:
            updated_expected.append(0)
        else:
            print "Inproper value in expected class values"

    return updated_predicted, updated_expected

def run_two_naive_bayes(filename, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs):


    """Take the train and test data, split it by gender, and train the two naive bayes classifiers
    """

    protected_train, y_protected_train, unprotected_train, y_unprotected_train = splitDataBySensitiveFeature(x_train, x_control_train[sensitive_attrs], y_train, sensitive_attrs)
    protected_test, y_protected_test, unprotected_test, y_unprotected_test = splitDataBySensitiveFeature(x_test, x_control_test[sensitive_attrs], y_test, sensitive_attrs)

    women_predicted_class_status, women_expected_class_status = predict(protected_train, y_protected_train, protected_test, y_protected_test)
    men_predicted_class_status, men_expected_class_status     = predict(unprotected_train, y_unprotected_train, unprotected_test, y_unprotected_test)

# 3 columns:
# Correct Class, Estimated Class, Sensitive Variable,


    f = open("RESULTS/"+filename, 'w')
    for i in range(0, len(women_predicted_class_status)-1):
	    line_of_data = ( str(women_expected_class_status[i]) + " " + str(women_predicted_class_status[i]) + " 0.0")
	    f.write(line_of_data)
	    f.write("\n")
    for i in range(0, len(men_predicted_class_status)):
	    line_of_data = ( str(men_expected_class_status[i]) + " " + str(men_predicted_class_status[i]) + " 1.0")
	    f.write(line_of_data)
	    f.write("\n")
    f.close()


    return women_predicted_class_status, women_expected_class_status, men_predicted_class_status, men_expected_class_status
