import os,sys
from subprocess import call
from sklearn.svm import SVC
from repairers import *
def run_audit():


    #First check if the data has already been repaired
    path =  os.getcwd()
    path = path+'/data/adult'
    data_files = os.listdir(path)

    if "repaired_adult.csv" not in data_files:
        print "Repairing adult data"
        bash_call = "python repairers/repair.py data/adult/adult.csv data/adult/repaired_adult.csv .1 -p income-per-year -i race"
        os.system(bash_call)


def svm_classify(filename, sensitive_attr, x_train, y_train, x_control_train, x_test, y_test, x_control_test):

    clf = SVC()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    """
    How Kamashima takes data:
    1 = Male (non-sensitive), 0 = Female (sensitive) in data
    3 columns:
    Correct Class, Estimated Class, Sensitive Variable
    How data comes from Blackbox/feldmen code:
    Pre-Repaired Feature, Response, Prediction
    """
    f = open("RESULTS/"+filename, 'w')
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
