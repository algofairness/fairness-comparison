import csv

def load_data():
  headers = ["Outcome", "X", "Y"]
  # Data from: https://github.com/jasonbaldridge/try-tf

  types = [str, float, float]
  with open("test_data/linear_data_train.csv") as f:
    reader = csv.reader(f)
    train = [[types[i](float(e)*100) for i, e in enumerate(row)] for row in reader]

  with open("test_data/linear_data_eval.csv") as f:
    reader = csv.reader(f)
    test = [[types[i](float(e)*100) for i, e in enumerate(row)] for row in reader]

  return headers, train, test

def test():
  headers, train, test = load_data()
  print "load_data -- unpacks correctly? -- ", (headers != None and train != None and test != None)
  print "Needs better test", False

if __name__=="__main__":
  test()


