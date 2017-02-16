from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import urllib2



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

def splitDataByGender(X, x_control, y):

    """
    X is a list of individuals, x_control is a list of each individual's sex (as binary variable),
    y is a list of that individual's actual class status as binary variable (where 0 = income <=50k, 1 = income >50k)
    X[i], x_control[i], y[i] should all correspond to the same individual. Accordingly each list should be of same length
    """

    #Converting from numpy list to standard python list
    X.tolist()
    x_control.tolist()
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

        if (x_control[i] == 0):
            women.append(person)
            y_women.append(y[i])
        else:
            men.append(person)
            y_men.append(y[i])

    return women, y_women, men, y_men


def predict(X, y, X_test, y_test):

    """
    classification report and confusion_matrix info
    -1 is <=50K
    1 is >50k
    Thus in binary classification, the count of true negatives is C_{0,0},
    false negatives is C_{1,0}, true positives is C_{1,1} and false positives
    is C_{0,1}.
    """

    # fit a Naive Bayes model to the data
    model = GaussianNB()

    # fit a Naive Bayes model to the data
    model = GaussianNB()
    model.fit(X, y)

    # make predictions
    expected = y_test
    predicted = model.predict(X_test)

    #summarize the fit of the model
    #print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))

    return predicted, expected

def run_two_naive_bayes(x_train, y_train, x_control_train, x_test, y_test, x_control_test):

    """Take the train and test data, split it by gender, and train the two naive bayes classifiers
    """

    women_train, y_women_train, men_train, y_men_train = splitDataByGender(x_train, x_control_train["sex"], y_train)
    women_test, y_women_test, men_test, y_men_test = splitDataByGender(x_test, x_control_test["sex"], y_test)

    #This unmodified naive bayes classifier is included for comparison
    predict (x_train, y_train, x_test, y_test)

    women_predicted_class_status, women_expected_class_status = predict(women_train, y_women_train, women_test, y_women_test)
    men_predicted_class_status, men_expected_class_status =     predict(men_train, y_men_train, men_test, y_men_test)

    return women_predicted_class_status, women_expected_class_status, men_predicted_class_status, men_expected_class_status
