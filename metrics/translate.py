def bin_map(col, pos, neg):
  res = []
  for x in col:
    if x == pos:
      res.append(1)
    if x == neg:
      res.append(0)
  return res

def get_metric_input(y_actual, y_predicted, protected, y_pos, y_neg, p_pos, p_neg):
  bin_actual = bin_map(y_actual, y_pos, y_neg)
  bin_predicted = bin_map(y_predicted, y_pos, y_neg)
  bin_protected = bin_map(protected, p_pos, p_neg)
  return bin_actual, bin_predicted, bin_protected 
