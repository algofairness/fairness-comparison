from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
import urllib2

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


    score = model.score(X_test, y_test)

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

def run_two_naive_bayes(gamma, filename, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs):


    """Take the train and test data, split it by gender, and train the two naive bayes classifiers
    """
    total_positive = 0
    total = len(y_test)
    for i in y_test:
        if i == 1:
            total_positive+=1
    #print total_positive
    #print "Total positive"

    protected_train, y_protected_train, unprotected_train, y_unprotected_train = splitDataBySensitiveFeature(x_train, x_control_train[sensitive_attrs], y_train, sensitive_attrs)

    protected_test, y_protected_test, unprotected_test, y_unprotected_test = splitDataBySensitiveFeature(x_test, x_control_test[sensitive_attrs], y_test, sensitive_attrs)

    #Train model
    women_predicted_class_status, women_expected_class_status = predict(protected_train, y_protected_train, protected_test, y_protected_test)
    men_predicted_class_status, men_expected_class_status     = predict(unprotected_train, y_unprotected_train, unprotected_test, y_unprotected_test)


    women_total = len(women_expected_class_status)
    men_total = len(men_expected_class_status)
    women_negative = 0
    women_positive = 0
    for i in women_predicted_class_status:
        if i == 0:
            women_negative+=1
        else:
            women_positive+=1

    #print women_positive
    #print women_negative

    men_negative = 0
    men_positive = 0
    for i in men_predicted_class_status:
        if i == 0:
            men_negative+=1
        else:
            men_positive+=1


    discrimination = 100.0
    while discrimination > gamma:
        #C+ S- is Women Positive
        #C- S- is Women Negative
        #C- S+ is Men Negative
        #C+ S+ is Men Positive
    #    print "men negative %f" % men_negative
    #    print "women_positive %f" % women_positive

        if men_positive+women_positive < total_positive:

    #        print "increasing protected group"

            goal_women_positive = women_positive + .01*men_negative
            goal_women_negative = women_negative - .01*men_negative

    #        print "Women positive %f" % women_positive
    #        print "Goal women positive %f" % goal_women_positive
            for i in range(0, len(y_protected_train)):
                if (goal_women_positive - women_positive) > 0:
                    # Whether the negative class is representred as -1.0 or 0.0 is an ongoing headache throughout this project
                    if (y_protected_train[i] == 0.0 or y_protected_train[i] == -1.0):
                        y_protected_train[i] = 1.0
                        women_positive +=1
                    else:
                        pass
                else:
                    break

    #        print "Women negative %f" % women_negative
    #        print "GWN %f" % goal_women_negative
            for i in range(0, len(y_protected_train)):
                if (women_negative - goal_women_negative) > 0:
                    if y_protected_train[i] == 0.0 or y_protected_train[i] == -1.0:
                        y_protected_train[i] = 1.0
                        women_negative -=1
                    else:
                        pass
                else:
                    break

            women_positive_count = 0

            for j in y_protected_train:
                if j == 1.0:
                    women_positive_count+=1
    #        print "Women positive count %f" % women_positive_count

        else:
    #        print "Decreasing unprotected group"
            """
            """
            goal_men_negative = men_negative + .01*women_positive
            goal_men_positive = men_positive - .01*women_positive

    #        print "\n"
    #        print men_positive
    #        print goal_men_positive
    #        print "\n"
    #        print men_negative
    #        print goal_men_negative
    #        print "\n"

    #        print men_positive - goal_men_positive

            for i in range(0, len(y_unprotected_train)):
                if (men_positive - goal_men_positive) > 0:
                    if (y_unprotected_train[i] == 1.0):
                        y_unprotected_train[i] = 0.0
                        men_positive -=1
                    else:
                        pass
                else:
                    break
            #big - small
            for i in range(0, len(y_unprotected_train)):
                if (goal_men_negative - men_negative) > 0:
                    if y_unprotected_train[i] == 1.0:
                        y_unprotected_train[i] = 0.0
                        goal_men_negative -=1
                    else:
                        pass
                else:
                    break

            men_positive_count = 0
            for j in y_unprotected_train:
                if j == 1.0:
                    men_positive_count+=1
            #print "Men positive count %f" % men_positive_count
        #Train model
        women_predicted_class_status, women_expected_class_status = predict(protected_train, y_protected_train, protected_test, y_protected_test)
        men_predicted_class_status, men_expected_class_status     = predict(unprotected_train, y_unprotected_train, unprotected_test, y_unprotected_test)

        women_negative = 0
        women_positive = 0
        for i in women_predicted_class_status:
            if i == 0:
                women_negative+=1
            else:
                women_positive+=1

        men_negative = 0
        men_positive = 0
        for i in men_predicted_class_status:
            if i == 0:
                men_negative+=1
            else:
                men_positive+=1


        #print "Men positive %f" % men_positive
        #print
        women_ratio = float(women_positive)/float(women_total)
        men_ratio = float(men_positive)/float(men_total)

        #print women_ratio
        #print men_ratio

        discrimination = men_ratio - women_ratio
        #print "Discrimination: %f" % discrimination

    f = open("algorithms/kamishima/00RESULT/"+filename, 'w')
    for i in range(0, len(women_predicted_class_status)):
        line_of_data = ( str(women_expected_class_status[i]) + " " + str(women_predicted_class_status[i]) + " 0.0")
        f.write(line_of_data)
        f.write("\n")
    for i in range(0, len(men_predicted_class_status)):
        line_of_data = ( str(men_expected_class_status[i]) + " " + str(men_predicted_class_status[i]) + " 1.0")
        f.write(line_of_data)
        f.write("\n")
    f.close()




# 3 columns:
# Correct Class, Estimated Class, Sensitive Variable,





    return women_predicted_class_status, women_expected_class_status, men_predicted_class_status, men_expected_class_status
