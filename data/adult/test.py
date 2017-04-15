# new_lines = []
# count = 0
# for line in open("adult.csv"):
#     line = line.strip()
#     count +=1
#     if line == "": continue # skip empty lines
#     if line[0] == "a": continue # skip line of feature categories, in csv
#     line = line.split(",")
#     if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
#         continue
#     else:
#         new_lines.append(line)
#
#
# f = open("adult-?.csv", 'w')
# for i in new_lines:
#     """
#     Convert -1 to 0 for Kamashima's classifiers
#     """
#     x = ""
#     for j in i:
#         x += str(j)
#         x +=","
#     x = x[:-1]
#     f.write(x)
#     f.write('\n')
# f.close()


new_lines = []
for line in open("adult-?-new-columns-nixed.csv"):
    line = line.strip()
    if line == "": continue # skip empty lines
    if line[0] == "a": continue # skip line of feature categories, in csv
    line = line.split(",")
    if str(line[3]) == "White":
        line[3] = 0
    else:
        line[3] = 1
    new_lines.append(line)


f = open("adult-?-new-columns-no.csv", 'w')
for i in new_lines:
    """
    Convert -1 to 0 for Kamashima's classifiers
    """
    x = ""
    for j in i:
        x += str(j)
        x +=","
    x = x[:-1]
    f.write(x)
    f.write('\n')
f.close()
