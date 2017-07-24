from sklearn.metrics import accuracy_score

class Metrics:
  # protected = 0, nonprotected = 1
  def __init__(self, actual, predicted, protected, DBC):
    self.actual = actual
    self.predicted = predicted
    self.protected = protected
    self.DBC = DBC
    self.total_sensitive = 0
    self.total_nonsensitive = 0
    for x in self.protected:
      if x == 0:
        self.total_sensitive += 1
      elif x == 1:
        self.total_nonsensitive += 1

  def accuracy(self):
    res = accuracy_score(self.actual, self.predicted)
    return res

  def DI_score(self):
    # otherwise known as the p-rule (Zafar)
    n_protected_pos = 0 
    n_nonprotected_pos = 0
    for i, x in enumerate(self.predicted):
      if x == 1 and self.protected[i] == 0:
        n_protected_pos += 1
      if x == 1 and self.protected[i] == 1:
        n_nonprotected_pos += 1 
    if (n_protected_pos == 0) or (n_nonprotected_pos == 0):
      return 'NA'
    if (self.total_sensitive == 0) or (self.total_nonsensitive == 0):
      return 'NA'
    p_protected_pos = n_protected_pos / float(self.total_sensitive)
    p_nonprotected_pos = n_nonprotected_pos / float(self.total_nonsensitive)
    res = (p_protected_pos / float(p_nonprotected_pos))
    return res

  def BER(self):
    n_neg_nonprotected = 0
    n_pos_protected = 0
    for i, x in enumerate(self.predicted):
      if x == 0 and self.protected[i] == 0:
        n_neg_nonprotected += 1
      if x == 1 and self.protected[i] == 1:
        n_pos_protected += 1
    p_neg_nonprotected = n_neg_nonprotected / float(len(self.predicted))
    p_pos_protected = n_pos_protected / float(len(self.predicted))
    p = p_neg_nonprotected + p_pos_protected
    res = p / 2.0
    return res

  def BER_derek(self):
    n_neg_nonprotected = 0
    n_pos_protected = 0
    for i, x in enumerate(self.predicted):
      if x == 0 and self.protected[i] == 0:
	n_neg_nonprotected += 1
      if x == 1 and self.protected[i] == 1:
	n_pos_protected += 1
    p_neg_nonprotected = n_neg_nonprotected / float(len(self.predicted))
    p_pos_protected = n_pos_protected / float(len(self.predicted)) 
    p = p_neg_nonprotected - p_pos_protected
    res = float((1 + p) / 2)
    return res

  def BCR(self):
    if self.BER() == 'NA':
      return 'NA'
    res = 1 - self.BER()
    return res

  def CV_score(self):
  # Calders-Verwer score
    n_protected_pos = 0
    n_nonprotected_pos = 0
    for i, x in enumerate(self.predicted):
      if x == 1 and self.protected[i] == 0:
        n_protected_pos += 1
      if x == 1 and self.protected[i] == 1:
        n_nonprotected_pos += 1
    if (self.total_sensitive == 0) or (self.total_nonsensitive == 0):
      return 'NA'
    p_protected_pos = n_protected_pos / float(self.total_sensitive)
    p_nonprotected_pos = n_nonprotected_pos / float(self.total_nonsensitive)
    res = p_protected_pos - p_nonprotected_pos
    return res

  def NPI_score(self):
  # Normalized prejudice index score
  # Evan's paper, section 2.10
    return

  def DBC_score(self):
  # Distance boundary covariance
  # Only works for Zafar, so can't use to compare?
    if self.DBC == None:
      self.DBC = 'NA'
    return self.DBC

  def DM_score(self):
  # Disparate mistreatment score
    return
    

if __name__=='__main__':
  a = [1,1,0,0]
  p = [1,0,1,0] 
  prot = [1,0,1,0]

  m = Metrics(a,p,prot)
  print(m.accuracy())
  print(m.DI_score())
  print(m.BER())
  print(m.BCR())
  print(m.DBC_score())
  print(m.CV_score())
