from copy import deepcopy

def get_median(values):
  """
  Given an unsorted list of numeric values, return median value (as a float).
  Note that in the case of even-length lists of values, we apply the value to
  the left of the center to be the median (such that the median can only be
  a value from the list of values).
  Eg: get_median([1,2,3,4]) == 2, not 2.5.
  """

  if not values:
    raise Exception("Cannot calculate median of list with no values!")

  sorted_values = deepcopy(values)
  sorted_values.sort() # Not calling `sorted` b/c `sorted_values` may not be list.

  if len(values) % 2 == 0:
    return sorted_values[len(values)/2-1]
  else:
    return sorted_values[len(values)/2]


def test():
  test_median()

def test_median():
  feature_values = [4,1,3,2]
  correct_median = 2
  print "median value is correct?", get_median(feature_values) == correct_median


if __name__=="__main__":
  test()
