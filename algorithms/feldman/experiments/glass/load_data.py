from splitters import split_by_percent
import csv
import random

# data from: http://archive.ics.uci.edu/ml/datasets/Glass+Identification
# id feature should be ignored

train_percentage = 1.0/2.0
filename = "test_data/glass.csv"
max_entries = None
correct_types = [int, float, float, float, float, float,float,
                 float,float,float,str]

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

  print "load_data types are correct? -- ", gathered_types == correct_types
  print "all headers given types? -- ", len(headers) == len(correct_types)


if __name__=="__main__":
  test()


