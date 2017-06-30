import random

def split_by_percent(data, train_percentage):
  train_indices = random.sample(range(len(data)), int(train_percentage*len(data)))
  train = [row for i,row in enumerate(data) if i in train_indices]
  test = [row for i,row in enumerate(data) if i not in train_indices]
  return train, test

def test():
  data = [[i] for i in range(100)]

  train,test = split_by_percent(data, 0.75)
  print "75:25 data split correct?", len(train) == 75 and len(test) == 25

  train,test = split_by_percent(data, 0.50)
  print "50:50 data split correct?", len(train) == 50 and len(test) == 50

if __name__=="__main__":
  test()

