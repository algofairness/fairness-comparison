import os,sys
from subprocess import call
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from repairers import *


"""
Should be able to make a single classify function, for standard NB/LR/SVM.
However, need to standardize positive/negative classification outcomes as 1/0,
So using a seperate, repetetive function per dataset right now.
"""

def run_compas_repair():

    #First check if the data has already been repaired
    path =  os.getcwd()
    path = path+'/data/propublica'
    data_files = os.listdir(path)

    if "repaired-compas-scores-two-years-violent-columns-removed_.8.csv" not in data_files:
        print "Repairing compas data"
        bash_call = "python repair.py data/propublica/compas-scores-two-years-violent-columns-removed.csv data/propublica/repaired-compas-scores-two-years-violent-columns-removed_.8.csv .8 -p race -i is_violent_recid id days_b_screening_arrest"
        os.system(bash_call)
        bash_call = "python repair.py data/propublica/compas-scores-two-years-violent-columns-removed.csv data/propublica/repaired-compas-scores-two-years-violent-columns-removed_.9.csv .9 -p race -i is_violent_recid id days_b_screening_arrest"
        os.system(bash_call)
        bash_call = "python repair.py data/propublica/compas-scores-two-years-violent-columns-removed.csv data/propublica/repaired-compas-scores-two-years-violent-columns-removed_1.csv 1 -p race -i is_violent_recid id days_b_screening_arrest"
        os.system(bash_call)
        print "Complete"

def run_german_repair():

    #First check if the data has already been repaired
    path =  os.getcwd()
    path = path+'/data/german'
    data_files = os.listdir(path)

    if "repaired_german_credit_data_.8.csv" not in data_files:
        print "Repairing German data"
        bash_call = "python repair.py data/german/german_credit_data.csv data/german/repaired_german_credit_data_.8.csv .8 -p personal_status"
        os.system(bash_call)
        bash_call = "python repair.py data/german/german_credit_data.csv data/german/repaired_german_credit_data_.9.csv .9 -p personal_status"
        os.system(bash_call)
        bash_call = "python repair.py data/german/german_credit_data.csv data/german/repaired_german_credit_data_1.csv 1 -p personal_status"
        os.system(bash_call)
        print "Repair Complete"


def run_adult_repair():

    #First check if the data has already been repaired
    path =  os.getcwd()
    path = path+'/data/adult'
    data_files = os.listdir(path)

    if "repaired_adult_.8.csv" not in data_files:
        path =  os.getcwd()
        print path
        print "Repairing adult data"
        bash_call = "python repair.py data/adult/adult.csv data/adult/repaired_adult_.8.csv .8 -p race -i income-per-year race"
        os.system(bash_call)
        bash_call = "python repair.py data/adult/adult.csv data/adult/repaired_adult_.9.csv .9 -p race -i income-per-year race"
        os.system(bash_call)
        bash_call = "python repair.py data/adult/adult.csv data/adult/repaired_adult_1.csv 1 -p race -i income-per-year race"
        os.system(bash_call)


        # bash_call = "python repair.py data/adult/adult.csv data/adult/sex_repaired_adult_.8.csv .8 -p sex -i race"
        # os.system(bash_call)
        # bash_call = "python repair.py data/adult/adult.csv data/adult/sex_repaired_adult_.9.csv .9 -p sex -i race"
        # os.system(bash_call)
        # bash_call = "python repair.py data/adult/adult.csv data/adult/sex_repaired_adult_1.csv 1 -p sex -i race"
        # os.system(bash_call)


def classify_adult(filename, sensitive_attr, x_train, y_train, x_control_train, x_test, y_test, x_control_test):

    clf = SVC()


    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    score = clf.score(x_test, y_test)

    print "LR Score+"+filename
    print score
    print "\n"


    """
    How Kamashima takes data:
    1 = Male (non-sensitive), 0 = Female (sensitive) in data
    3 columns:
    Correct Class, Estimated Class, Sensitive Variable
    How data comes from Blackbox/feldmen code:
    Pre-Repaired Feature, Response, Prediction
    """
    f = open("RESULTS/svm+"+filename, 'w')
    new_predictions = []
    new_y_test = []

    for j in range(0, len(predictions)):
        if predictions[j] == -1.:
            new_predictions.append(0)
        elif predictions[j] == 1.:
            new_predictions.append(1)

    for j in range(0, len(y_test)):
        if y_test[j] == -1.:
            new_y_test.append(0)
        elif y_test[j] == 1.:
            new_y_test.append(1)


    for i in range(0, len(x_test)):
        string = (str(new_y_test[int(i)])+" " + str(new_predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()


    nb = GaussianNB()
    nb.fit(x_train, y_train)
    predictions = nb.predict(x_test)
    score = nb.score(x_test, y_test)

    print "NB Score+"+filename
    print score
    print "\n"

    f = open("RESULTS/nb+"+filename, 'w')
    new_predictions = []
    new_y_test = []

    for j in range(0, len(predictions)):
        if predictions[j] == -1.:
            new_predictions.append(0)
        elif predictions[j] == 1.:
            new_predictions.append(1)

    for j in range(0, len(y_test)):
        if y_test[j] == -1.:
            new_y_test.append(0)
        elif y_test[j] == 1.:
            new_y_test.append(1)


    for i in range(0, len(x_test)):
        """
        Convert -1 to 0 for Kamashima's classifiers
        """
        string = (str(new_y_test[int(i)])+" " + str(new_predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    score = lr.score(x_test, y_test)
    f = open("RESULTS/lr+"+filename, 'w')
    new_predictions = []
    new_y_test = []

    for j in range(0, len(predictions)):
        if predictions[j] == -1.:
            new_predictions.append(0)
        elif predictions[j] == 1.:
            new_predictions.append(1)

    for j in range(0, len(y_test)):
        if y_test[j] == -1.:
            new_y_test.append(0)
        elif y_test[j] == 1.:
            new_y_test.append(1)


    for i in range(0, len(x_test)):
        """
        Convert -1 to 0 for Kamashima's classifiers
        """
        string = (str(new_y_test[int(i)])+" " + str(new_predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()

def classify_german(filename, sensitive_attr, x_train, y_train, x_control_train, x_test, y_test, x_control_test):

    clf = SVC()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    score = clf.score(x_test, y_test)

    """
    How Kamashima takes data:
    1 = Male (non-sensitive), 0 = Female (sensitive) in data
    3 columns:
    Correct Class, Estimated Class, Sensitive Variable
    How data comes from Blackbox/feldmen code:
    Pre-Repaired Feature, Response, Prediction
    """
    f = open("RESULTS/svm+"+filename, 'w')
    new_predictions = []
    new_y_test = []

    for j in range(0, len(predictions)):
        if predictions[j] == 0.:
            new_predictions.append(0)
        elif predictions[j] == 1.:
            new_predictions.append(1)

    for j in range(0, len(y_test)):
        if y_test[j] == 0.:
            new_y_test.append(0)
        elif y_test[j] == 1.:
            new_y_test.append(1)


    for i in range(0, len(x_test)):
        string = (str(new_y_test[int(i)])+" " + str(new_predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()


    nb = GaussianNB()
    nb.fit(x_train, y_train)
    predictions = nb.predict(x_test)
    score = nb.score(x_test, y_test)

    f = open("RESULTS/nb+"+filename, 'w')
    new_predictions = []
    new_y_test = []

    for j in range(0, len(predictions)):
        if predictions[j] == 0.:
            new_predictions.append(0)
        elif predictions[j] == 1.:
            new_predictions.append(1)

    for j in range(0, len(y_test)):
        if y_test[j] == 0.:
            new_y_test.append(0)
        elif y_test[j] == 1.:
            new_y_test.append(1)


    for i in range(0, len(x_test)):
        """
        Convert floats to ints Kamashima's classifiers
        """
        string = (str(new_y_test[int(i)])+" " + str(new_predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    score = lr.score(x_test, y_test)

    f = open("RESULTS/lr+"+filename, 'w')
    new_predictions = []
    new_y_test = []

    for j in range(0, len(predictions)):
        if predictions[j] == 0.:
            new_predictions.append(0)
        elif predictions[j] == 1.:
            new_predictions.append(1)

    for j in range(0, len(y_test)):
        if y_test[j] == 0.:
            new_y_test.append(0)
        elif y_test[j] == 1.:
            new_y_test.append(1)


    assert(len(new_y_test) == len(y_test))
    assert(len(new_predictions) == len(predictions))

    for i in range(0, len(x_test)):
        """
        Convert floats to ints for Kamashima's classifier
        """
        string = (str(new_y_test[int(i)])+" " + str(new_predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()



def classify_compas(filename, sensitive_attr, x_train, y_train, x_control_train, x_test, y_test, x_control_test):

    clf = SVC()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    score = clf.score(x_test, y_test)

    print "LR Score+"+filename
    print score
    print "\n"

    """
    How Kamashima takes data:
    1 = Male (non-sensitive), 0 = Female (sensitive) in data
    3 columns:
    Correct Class, Estimated Class, Sensitive Variable
    How data comes from Blackbox/feldmen code:
    Pre-Repaired Feature, Response, Prediction
    """
    f = open("RESULTS/svm+"+filename, 'w')

    for i in range(0, len(x_test)):
        string = (str(y_test[int(i)])+" " + str(predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()


    nb = GaussianNB()
    nb.fit(x_train, y_train)
    predictions = nb.predict(x_test)
    score = nb.score(x_test, y_test)

    print "NB Score+"+filename
    print score
    print "\n"

    f = open("RESULTS/nb+"+filename, 'w')

    for i in range(0, len(x_test)):
        """
        Convert -1 to 0 for Kamashima's classifiers
        """
        string = (str(y_test[int(i)])+" " + str(predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    score = lr.score(x_test, y_test)
    f = open("RESULTS/lr+"+filename, 'w')

    for i in range(0, len(x_test)):
        string = (str(y_test[int(i)])+" " + str(predictions[i]) + " " +str(x_control_test[sensitive_attr][int(i)]))
        f.write(string)
        f.write('\n')
    f.close()
