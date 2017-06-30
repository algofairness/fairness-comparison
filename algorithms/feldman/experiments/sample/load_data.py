from splitters import split_by_percent
import random

train_percentage = 2.0/3.0

def load_data():
  N = 6000
  headers = ["Feature A (i)", "Feature B (2i)", "Feature C (-i)",
             "Constant Feature", "Random Feature", "Outcome"]

  data = [[i, 2*i, -i, 1, random.random(), "A"] for i in range(0,N/2)] + \
          [[i, 2*i, -i, 1, random.random(), "B"] for i in range(N/2,N)]

  train, test = split_by_percent(data, train_percentage)

  return headers, train, test

def test():
  headers, train, test = load_data()
  print "load_data -- unpacks correctly? -- ", (headers != None and train != None and test != None)

if __name__=="__main__":
  test()


