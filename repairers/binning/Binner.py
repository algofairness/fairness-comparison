from BinSizes import FreedmanDiaconisBinSize as bsc

def make_histogram_bins(bin_size_calculator, data, col_id):
  feature_vals = [row[col_id] for row in data]
  bin_range = bin_size_calculator(feature_vals)

  if bin_range==0.0:
    bin_range = 1.0

  data_tuples = list(enumerate(data)) # [(0,row), (1,row'), (2,row''), ... ]
  sorted_data_tuples = sorted(data_tuples, key=lambda tup: tup[1][col_id])

  max_val = max(data, key=lambda datum: datum[col_id])[col_id]
  min_val = min(data, key=lambda datum: datum[col_id])[col_id]

  index_bins = []
  val_ranges = []
  curr = min_val
  while curr <= max_val:
    index_bins.append([])
    val_ranges.append((curr, curr+bin_range))
    curr += bin_range

  for row_index, row in sorted_data_tuples:
    for bin_num, val_range in enumerate(val_ranges):
      if val_range[0] <= row[col_id] < val_range[1]:
        index_bins[bin_num].append(row_index)
        break

  index_bins = [b for b in index_bins if b]

  return index_bins


def test():
  data = [[i,0] for i in xrange(0, 100)]
  bins = make_histogram_bins(bsc, data, 0)
  print "make_histogram_bins -- no entries lost --", sum(len(row) for row in bins) == len(data)
  print "make_histogram_bins -- correct # of bins --", (len(bins) == 5)

  data = [[1]]*100
  bins = make_histogram_bins(bsc, data, 0)
  print "homogenous feature yields one bin? ", len(bins) == 1

  data = [[1]]*100 + [[2]]
  bins = make_histogram_bins(bsc, data, 0)
  print "bins being bucketed by value? -- ", len(bins) == 2



if __name__=="__main__":
  test()
