from splitters import split_by_percent
import csv
import random

train_percentage = 2.0/3.0
filename = "test_data/german_categorical.csv"
max_entries = None
correct_types = [str, int, str, str, int, str,str,
                 int,str,str,int, str, int, str, str, int,
                 str, int, str, str, str]

AGE_COL = 12

def load_data():
  with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]
    headers = data.pop(0)

    if max_entries:
      data = random.sample(data, max_entries)

    for i, row in enumerate(data):
      for j, correct_type in enumerate(correct_types):
        data[i][j] = correct_type(row[j])

      # Replace the numeric age with "young" and "old" categories.
      # Threshold based on: F. Kamiran and T. Calders. Classifying without discriminating.
      data[i][AGE_COL] = "old" if row[AGE_COL] > 25 else "young"

    train, test = split_by_percent(data, train_percentage)

  return headers, train, test


def test():
  headers, train, test = load_data()
  print "load_data unpacks correctly? -- ", (headers != None and train != None and test != None)

  gathered_types = []
  for i, header in enumerate(headers):
    if all( isinstance(row[i],float) for row in train + test ):
      gathered_types.append(float)
    elif all( isinstance(row[i],int) for row in train + test ):
      gathered_types.append(int)
    elif all( isinstance(row[i],str) for row in train + test ):
      gathered_types.append(str)

  # The below test is supposed to fail because the age column type changes from the
  # read in correct type of int to the old/young type of str
  # print "load_data types are correct? -- ", gathered_types == correct_types
  print "all headers given types? -- ", len(headers) == len(correct_types)


if __name__=="__main__":
  test()


