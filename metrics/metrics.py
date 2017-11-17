from sklearn.metrics import accuracy_score, auc, matthews_corrcoef, average_precision_score
from algorithms.kamishima.fadm.eval._bin_class import BinClassStats

class Metrics:
  # protected = 0, nonprotected = 1
  def __init__(self, actual, predicted, protected):
    self.actual = actual
    self.predicted = predicted
    self.protected = protected
    self.total_sensitive = 0
    self.total_nonsensitive = 0
    for x in self.protected:
      if x == 0:
        self.total_sensitive += 1
      elif x == 1:
        self.total_nonsensitive += 1
    self.tp = 0
    self.fp = 0
    self.tn = 0
    self.fn = 0
    for i in range(len(self.predicted)):
      if self.actual[i] == 1 and self.predicted[i] == 1:
        self.tp += 1
      if self.actual[i] == 0 and self.predicted[i] == 0:
        self.tn += 1
      if self.actual[i] == 1 and self.predicted[i] == 0:
        self.fn += 1
      if self.actual[i] == 0 and self.predicted[i] == 1:
        self.fp += 1

  def accuracy(self):
    res = accuracy_score(self.actual, self.predicted)
    return res

  def APS(self):
    res = average_precision_score(self.actual, self.predicted)
    return res

  def AUC(self):
    #WRONG
    # Using Trapezoidal Rule
    res = auc(self.actual, self.predicted)
    return res

  def MCC(self):
    res = matthews_corrcoef(self.actual,self.predicted, sample_weight=None)
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

  # Still unable to replicate Evan's results
  def BER(self):
    if(len(self.predicted) == 0):
      return "NA"
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

  # Still unable to replicate Evan's results
  def BCR(self):
    if self.BER() == 'NA':
      return 'NA'
    res = 1 - self.BER()
    return res

  # Still unable to replicate Evan's results
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
    return 1-res

  def NPI_score_2(self):
    stats = BinClassStats(self.tp, self.fn, self.fp, self.tn)
    mi, nmic, nmie, amean, gmean = stats.mi2()
    return gmean

  # Still unanble to replicate Evan's results (correct calculation of NPI?)
  def NPI_score_nat(self):
  # Normalized prejudice index score
  # Evan's paper, section 2.10
    stats = BinClassStats(self.tp, self.fn, self.fp, self.tn)
    mi, nmic, nmie, amean, gmean = stats.mi()
    return gmean

#  def DBC_score(self):
  # Distance boundary covariance
  # Only works for Zafar, so can't use to compare?
#  distances_boundary_test = np.dot(x_test, lr.coef_[0])
#  cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, [sensitive_attr])
#  DBC = ut.DBC([cov_dict_test], str(sensitive_attr))

#  distances_boundary_test = (np.dot(x_test, w)).tolist()
#  predictions = np.sign(distances_boundary_test)
#  cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, [sensitive_attrs[0]])
#  DBC = ut.DBC([cov_dict_test], sensitive_attrs[0])

#    if self.DBC == None:
#      self.DBC = 'NA'
#    return self.DBC

#  def DM_score(self):
#  # Disparate mistreatment score
#    return
    

if __name__=='__main__':
  a = [1,1,0,0]
  p = [1,0,1,0] 
  prot = [1,0,1,0]

  m = Metrics(a,p,prot)
  print((m.accuracy()))
  print((m.DI_score()))
  print((m.BER()))
  print((m.BCR()))
# print(m.DBC_score())
  print((m.CV_score()))
# print((m.NPI_score_nat()))
