from sklearn.metrics import accuracy_score

class Metrics:
  # protected = 0, nonprotected = 1
  def __init__(self, actual, predicted, protected):
    self.actual = actual
    self.predicted = predicted
    self.protected = protected

  def accuracy(self):
    res = accuracy_score(self.actual, self.predicted)
    return res

  def DI_score(self):
    # otherwise known as the p-rule (Zafar)
    n_protected_pos = 0 
    n_nonprotected_pos = 0
    for i, x in enumerate(self.actual):
      if x == 1 and self.protected[i] == 0:
        n_protected_pos += 1
      if x == 1 and self.protected[i] == 1:
        n_nonprotected_pos += 1 
    p_protected_pos = n_protected_pos / float(len(self.actual))
    p_nonprotected_pos = n_nonprotected_pos / float(len(self.actual))
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
    res = (p_neg_nonprotected / float(p_pos_protected))
    return res

  def utility(self):
    res = 1 - self.BER()
    return res

  def CV_score(self):
  # Calders-Verwer score
    n_protected_pos = 0
    n_nonprotected_pos = 0
    for i, x in enumerate(self.actual):
      if x == 1 and self.protected[i] == 0:
        n_protected_pos += 1
      if x == 1 and self.protected[i] == 1:
        n_nonprotected_pos += 1
    p_protected_pos = n_protected_pos / float(len(self.actual))
    p_nonprotected_pos = n_nonprotected_pos / float(len(self.actual))
    res = p_protected_pos - p_nonprotected_pos
    return res

  def NPI_score(self):
  # Normalized prejudice index score
  # Evan's paper, section 2.10
    return

  def DBC_score(self):
  # Distance boundary covariance
  # Only works for Zafar, so can't use to compare?
    return

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
  print(m.utility())
  print(m.CV_score())
