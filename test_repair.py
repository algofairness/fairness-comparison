import seaborn as sns, numpy as np
import matplotlib.pyplot as plt

f = open("data/adult.data", 'r')
age_of_men_in_raw_data = []
age_of_women_in_raw_data = []

for line in f:
    array = line.split(',')
    if (array[9] == " Male"):
        if (array[7] == " Wife"):
            age_of_men_in_raw_data.append(1.0)
        elif (array[7] == " Own-child"):
            age_of_men_in_raw_data.append(2.0)
        elif (array[7] == " Husband"):
            age_of_men_in_raw_data.append(3.0)
        elif (array[7] == " Not-in-family"):
            age_of_men_in_raw_data.append(4.0)
        elif (array[7] == " Other-relative"):
            age_of_men_in_raw_data.append(5.0)
        elif (array[7] == " Unmarried"):
            age_of_men_in_raw_data.append(6.0)

    if (array[9] == " Female"):
        if (array[7] == " Wife"):
            age_of_women_in_raw_data.append(1.0)
        elif (array[7] == " Own-child"):
            age_of_women_in_raw_data.append(2.0)
        elif (array[7] == " Husband"):
            age_of_women_in_raw_data.append(3.0)
        elif (array[7] == " Not-in-family"):
            age_of_women_in_raw_data.append(4.0)
        elif (array[7] == " Other-relative"):
            age_of_women_in_raw_data.append(5.0)
        elif (array[7] == " Unmarried"):
            age_of_women_in_raw_data.append(6.0)

f.close()


f = open("data/repair_new.csv", 'r')
age_of_men_in_repair_data = []
age_of_women_in_repair_data = []

for line in f:
    array = line.split(',')
    if (array[9] == "Male"):
        if (array[7] == "Wife"):

            age_of_men_in_repair_data.append(1.0)
        elif (array[7] == "Own-child"):
            age_of_men_in_repair_data.append(2.0)
        elif (array[7] == "Husband"):
            age_of_men_in_repair_data.append(3.0)
        elif (array[7] == "Not-in-family"):
            age_of_men_in_repair_data.append(4.0)
        elif (array[7] == "Other-relative"):
            age_of_men_in_repair_data.append(5.0)
        elif (array[7] == "Unmarried"):
            age_of_men_in_repair_data.append(6.0)

    if (array[9] == "Female"):
        if (array[7] == "Wife"):
            print "hey"
            age_of_women_in_repair_data.append(1.0)
        elif (array[7] == "Own-child"):
            age_of_women_in_repair_data.append(2.0)
        elif (array[7] == "Husband"):
            age_of_women_in_repair_data.append(3.0)
        elif (array[7] == "Not-in-family"):
            age_of_women_in_repair_data.append(4.0)
        elif (array[7] == "Other-relative"):
            age_of_women_in_repair_data.append(5.0)
        elif (array[7] == "Unmarried"):
            age_of_women_in_repair_data.append(6.0)

f.close()

age_of_men_in_raw_data = np.array(age_of_men_in_raw_data)
ax1 = sns.distplot(age_of_men_in_raw_data, color="g")
age_of_women_in_raw_data = np.array(age_of_women_in_raw_data).astype(np.float)
ax1 = sns.distplot(age_of_women_in_raw_data, color="m")
#plt.show()

age_of_men_in_repair_data = np.array(age_of_men_in_repair_data)
ax = sns.distplot(age_of_men_in_repair_data, color="r")
age_of_women_in_repair_data = np.array(age_of_women_in_repair_data)
ax = sns.distplot(age_of_women_in_repair_data, color="b")
plt.show()
