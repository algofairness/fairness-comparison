"""

NEED TO DIVE INTO FILE WHERE RESULTS ARE BEING CALCULATED TO DO THIS PROPERLY!!!

1   cond experimental condition
11  Acc  cccuracy
21  MI   mutual information between an estimated class and a sample class
25  NMI  normalized version of the column 22
47  PI   mutual information between an estimated class and a sensitive feature
51  NPI  normalized version of the column 47
55  UEI  UEI
56  SCVS Calders-Verwer score in terms of a sample class
57  ECVS Calders-Verwer score in terms of a estimated class


0   cond experimental condition
9  Acc  cccuracy
19  MI   mutual information between an estimated class and a sample class
23  NMI  normalized version of the column 22
45  PI   mutual information between an estimated class and a sensitive feature
49  NPI  normalized version of the column 47
53  UEI  UEI
54  SCVS Calders-Verwer score in terms of a sample class
55  ECVS Calders-Verwer score in terms of a estimated class
"""

methods = []
results = []
for line in open("adultd@t.txt", 'r'):
    if "method" in line:
        method = line
        methods.append(line)
    if (len(line) > 100):
        x = line.split()
        print method
        print "Accuracy: " + x[9]
        print "NPI: " + x[49]
        print "CVS: " + x[55]
        print "\n"
        results.append(x)



dictionary = dict(zip(methods, results))
#print dictionary
