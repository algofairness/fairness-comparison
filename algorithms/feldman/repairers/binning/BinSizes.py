import numpy

def FreedmanDiaconisBinSize(feature_values):
  """
  The bin size in FD-binning is given by size = 2 * IQR(x) * n^(-1/3)
  More Info: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule

  If the BinSize ends up being 0 (in the case that all values are the same),
  return a BinSize of 1.
  """

  q75, q25 = numpy.percentile(feature_values, [75, 25])
  IQR = q75 - q25

  return 2.0 * IQR * len(feature_values) ** (-1.0/3.0)


def test():
  values = range(0,100)
  bin_size = FreedmanDiaconisBinSize(values)
  correct_bin_size = 21
  bin_size = round(bin_size)
  print "FreedmanDiaconisBinSize -- correct size of bins? ", bin_size == correct_bin_size


if __name__=="__main__":
  test()
