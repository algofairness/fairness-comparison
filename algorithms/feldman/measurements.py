def accuracy(conf_matrix):
  """
  Given a confusion matrix, returns the accuracy.
  Accuracy Definition: http://research.ics.aalto.fi/events/eyechallenge2005/evaluation.shtml
  """
  total, correct = 0.0, 0.0
  for true_response, guess_dict in conf_matrix.items():
    for guess, count in guess_dict.items():
      if true_response == guess:
        correct += count
      total += count
  return correct/total

def BCR(conf_matrix):
  """
  Given a confusion matrix, returns Balanced Classification Rate.
  BCR is (1 - Balanced Error Rate).
  BER Definition: http://research.ics.aalto.fi/events/eyechallenge2005/evaluation.shtml
  """
  parts = []
  for true_response, guess_dict in conf_matrix.items():
    error = 0.0
    total = 0.0
    for guess, count in guess_dict.items():
      if true_response != guess:
        error += count
      total += count
    parts.append(error/total)
  BER = sum(parts)/len(parts)
  return 1 - BER

def get_conf_matrix(prediction_tuples):
  # Produce a confusion matrix in a dictionary format from those predictions.
  conf_table = {}
  for actual, guess in prediction_tuples:
    guess = convert_to_type(actual, guess) # ... Since file-reading changes types.

    if not actual in conf_table:
      conf_table[actual] = {}

    if not guess in conf_table[actual]:
      conf_table[actual][guess] = 1
    else:
      conf_table[actual][guess] += 1

  return conf_table

def convert_to_type(actual, guess):
  if type(actual) == bool:
    guess = guess=="True"
  else:
    actual_type = type(actual)
    guess = actual_type(guess)
  return guess


def test():
  test_conf_matrix()
  test_measurers()

def test_conf_matrix():
  pred_tuples = [(1,1),(1,1),(2,2),(3,3),(1,3),(3,1)]
  correct_conf_matrix = {1:{1:2, 3:1}, 2:{2:1}, 3:{1:1, 3:1}}
  conf_matrix = get_conf_matrix(pred_tuples)
  print "confusion matrix correct? -- ", conf_matrix == correct_conf_matrix

def test_measurers():
  conf_matrix = {"A":{"A":10}, "B":{"A":5,"B":5}}
  print "measurements -- accuracy correct? -- ", accuracy(conf_matrix) == 0.75
  print "measurements -- 1-BER correct? -- ", BCR(conf_matrix) == 0.75

  conf_matrix = {"A":{"A":10}}
  print "measurements -- accuracy correct? -- ", accuracy(conf_matrix) == 1.0
  print "measurements -- 1-BER correct? -- ", BCR(conf_matrix) == 1.0

  conf_matrix = {"A":{"A":95}, "B":{"A":5}}
  print "measurements -- accuracy correct? -- ", accuracy(conf_matrix) == 0.95
  print "measurements -- 1-BER correct? -- ", BCR(conf_matrix) == 0.50

if __name__=="__main__":
  test()
