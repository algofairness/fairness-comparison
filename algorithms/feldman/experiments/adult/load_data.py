import csv
import random

train_filename = "test_data/adult.csv"
test_filename = "test_data/adult.test.csv"
max_entries = None
correct_types = [int, str, int, str, int, str, str, str,
                 str,str, int, int, int, str, str]

def load_data():
  with open(train_filename) as f:
    reader = csv.reader(f)
    train = [row for row in reader]
    headers = train.pop(0)

    if max_entries:
      train = random.sample(train, max_entries/2)

    for i, row in enumerate(train):
      for j, correct_type in enumerate(correct_types):
        train[i][j] = correct_type(row[j])

  with open(test_filename) as f:
    reader = csv.reader(f)
    test = [row for row in reader][1:] # Ignore headers.

    if max_entries:
      test = random.sample(test, max_entries/2)

    for i, row in enumerate(test):
      for j, correct_type in enumerate(correct_types):
        test[i][j] = correct_type(row[j])

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
  print "load_data train size correct? -- ", len(train) == 32561
  print "load_data test size correct? -- ", len(test) == 16281

if __name__=="__main__":
  test()


