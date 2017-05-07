new_lines = []
count = 0

for line in open("german_numeric.csv"):
    line = line.strip()
    if line == "": continue # skip empty lines
    if line[0] == "O": continue # skip line of feature categories, in csv
    line = line.split(",")

    if (int(line[23]) == 1 or int(line[23]) == 3 or int(line[23]) == 4):
        line[23] = 1
    else:
        line[23] = 0

    if int(line[24]) == 2:
        line[24] = 0
        count +=1

    new_lines.append(line)
f = open("german_numeric_sex_encoded.csv", 'w')
for i in new_lines:

    x = ""
    for j in i:
        x += str(j)
        x +=","
    x = x[:-1]
    f.write(x)
    f.write('\n')
f.close()
