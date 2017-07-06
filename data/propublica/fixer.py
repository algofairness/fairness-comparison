new_lines = []
count = 0

#id,sex,age,age_cat,juv_fel_count,juv_misd_count,juv_other_count,priors_count,c_charge_degree,is_violent_recid,race,days_b_screening_arrest

for line in open("compas-scores-two-years-violent-columns-removed.csv"):
    line = line.strip()
    if line == "": continue # skip empty lines
    if line[0] == "i": continue # skip line of feature categories, in csv
    line = line.split(",")
    #Remove ID
    line =  line[1:]
    if line[0] == "Male":
        line[0] = 1
    if line[0] == "Female":
        line[0] = 0
    if line[2] == "Less than 25":
        line[2] = 0
    if line[2] == "25 - 45":
        line[2] = 1
    if line[2] == "Greater than 45":
        line[2] = 2

    if line[7] == "M":
        line[7] = 0
    if line[7] == "F":
        line[7] = 1

    if line[9] == "Caucasian":
        line[9] = 1
    if line[9] == "African-American":
        line[9] = 0
    if line[9] == "Hispanic" or line[9] == "Hispanic" or line[9] == "Native American" or line[9] == "Other" or line[9] == "Asian":
        line[9] = 3

    count +=1

    """
    Removing data here that is not to be used
    """
    if (line[10] != ""):
        var = int(line[10])
    else:
        var = 100

    if ((var > 30) or (int(line[9]) == 3) or (line[7] == "O") or (int(line[8]) == -1)):
        continue
    else:
        new_lines.append(line[0:-1])


f = open("compas-scores-violent-columns-removed-all-numeric.csv", 'w')
for i in new_lines:

    x = ""
    for j in i:
        x += str(j)
        x +=","
    x = x[:-1]
    f.write(x)
    f.write('\n')
f.close()
