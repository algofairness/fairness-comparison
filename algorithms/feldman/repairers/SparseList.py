class SparseList(list):
  def __init__(self, default=0, data=None):
    self.default = default
    self.vals = {}
    self.size = 0

    if data:
      self.extend(data)

  def __setitem__(self, index, value):
    if self.default != value:
      self.vals[index] = value
    self.size += 1

  def __len__(self):
    return self.size

  def __getitem__(self, index):
    if index in self.vals:
      return self.vals[index]
    else:
      return self.default

  def __repr__(self):
    return "<SparseList {}>".format(self.vals)

  def append(self, val):
    if self.default != val:
      self.vals[self.size] = val
    self.size += 1

  def extend(self, iterator):
    for val in iterator:
      if self.default != val:
        self.vals[self.size] = val
      self.size += 1

  def sort(self):
    values = sorted(self.vals.values())
    self.vals = {}
    old_size = self.size
    self.size = 0

    need_to_add_default = True
    for value in values:
      if need_to_add_default and self.default < value:
          self.size += (old_size - len(values))
          need_to_add_default = False

      if self.default != value:
        self.vals[self.size] = value
      self.size += 1

def audit_test():
  N=25000
  l = SparseList(default=0)
  for i in xrange(N):
    l.append(0)

  for i in xrange(N):
    l.append(i)

  for i in xrange(N):
    l.append(0)

  l.extend(xrange(N))

  l.sort()

  for i in xrange(4*N):
    l[i] # Call the `getter`

  print "Big SparseList size correct?", len(l) == 4*N

def test():
  l = SparseList(default=0)
  l.append(0)
  l.append(0)
  l.append(-1)
  l.append(2)

  print "SparseList size correct?", len(l) == 4
  print "SparseList accessed correctly?", l[0]==0 and l[2]==-1 and l[3] == 2

  l.sort()
  print "Sorted SparseList size correct?", len(l) == 4
  print "Sorted SparseList accessed correct?", l[0] == -1 and l[1]==0 and l[3] == 2

  l.append(100)
  l.append(-100)
  l.sort()
  print "Resorted SparseList size correct?", len(l) == 6
  print "Resorted SparseList accessed correct?",  l[0] == -100 and l[5]==100


if __name__=="__main__":
  test()
  audit_test()

