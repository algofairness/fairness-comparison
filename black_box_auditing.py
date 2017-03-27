def encode_blackbox_audit():
    f = open("sex.audit.repaired_0.8.predictions", 'r')
    """
    How Kamashima takes data:
    1 = Male (non-sensitive), 0 = Female (sensitive) in data
    3 columns:
    Correct Class, Estimated Class, Sensitive Variable

    How data comes from Blackbox/feldmen code:
    Pre-Repaired Feature, Response, Prediction

    """

    data = []
    for line in f:
        array = line.split()
        array = array[0].split(',')
        data.append(array)
    f.close()

    rearranged_blackbox_results = []

    for line in data[1:]:


        if line[0] == 'Male':
            line[0] = 1.0
        elif line[0] == "Female":
            line[0] = 0.0

        if line[1] == '>50K':
            line[1] = 1
        elif line[1] == '<=50K':
            line[1] = 0

        if line[2] == '>50K':
            line[2] = 1
        elif line[2] == '<=50K':
            line[2] = 0

        string = (str(line[1])+" " + str(line[2]) + " " +str(line[0]))
        #print string
        rearranged_blackbox_results.append(string)



    f = open("RESULTS/black_box_audit", 'w')
    for i in rearranged_blackbox_results:
        f.write(i)
        f.write('\n')
    f.close()
