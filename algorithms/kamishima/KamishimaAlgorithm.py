import sys
sys.path.append('/home/h205c/Derek/fairness-comparison')
from sklearn.metrics import accuracy_score
from algorithms.AbstractAlgorithm import *
from algorithms.kamishima.prejudice_regularizer import *
from datetime import datetime
import pandas as pd

class KamishimaAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(KamishimaAlgorithm, self).__init__(*args, **kwargs)
    if self.data == 'german':
      self.name = "german"

    if self.data == 'compas':
      self.name = "propublica"

    if self.data == 'adult':
      self.name = "sex_adult"

  def acc(actual, predicted):
    return accuracy_score(actual, predicted)

  def DI_score(predicted, protected,total_sensitive, total_nonsensitive):
    # otherwise known as the p-rule (Zafar)
    n_protected_pos = 0
    n_nonprotected_pos = 0
    for i, x in enumerate(predicted):
      if x == 1 and protected[i] == 0:
        n_protected_pos += 1
      if x == 1 and protected[i] == 1:
        n_nonprotected_pos += 1
    if (n_protected_pos == 0) or (n_nonprotected_pos == 0):
      return 'NA'
    if (total_sensitive == 0) or (total_nonsensitive == 0):
      return 'NA'
    p_protected_pos = n_protected_pos / float(total_sensitive)
    p_nonprotected_pos = n_nonprotected_pos / float(total_nonsensitive)
    return (p_protected_pos / float(p_nonprotected_pos))

  def CV_score(actual, predicted, total_sensitive, total_nonsensitive):
  # Calders-Verwer score
    n_protected_pos = 0
    n_nonprotected_pos = 0
    for i, x in enumerate(predicted):
      if x == 1 and protected[i] == 0:
        n_protected_pos += 1
      if x == 1 and protected[i] == 1:
        n_nonprotected_pos += 1
    if (total_sensitive == 0) or (total_nonsensitive == 0):
      return 'NA'
    p_protected_pos = n_protected_pos / float(total_sensitive)
    p_nonprotected_pos = n_nonprotected_pos / float(total_nonsensitive)
    return p_protected_pos - p_nonprotected_pos

  def run(self):

    def DI_score(predicted, protected,total_sensitive, total_nonsensitive):
      # otherwise known as the p-rule (Zafar)
      n_protected_pos = 0
      n_nonprotected_pos = 0
      for i, x in enumerate(predicted):
        if x == 1 and protected[i] == 0:
          n_protected_pos += 1
        if x == 1 and protected[i] == 1:
          n_nonprotected_pos += 1
      if (n_protected_pos == 0) or (n_nonprotected_pos == 0):
        return 999999999999 
      if (total_sensitive == 0) or (total_nonsensitive == 0):
        return 999999999999 
      p_protected_pos = n_protected_pos / float(total_sensitive)
      p_nonprotected_pos = n_nonprotected_pos / float(total_nonsensitive)
      return (p_protected_pos / float(p_nonprotected_pos))

    def CV_score(predicted, protected, total_sensitive, total_nonsensitive):
    # Calders-Verwer score
      n_protected_pos = 0
      n_nonprotected_pos = 0
      for i, x in enumerate(predicted):
        if x == 1 and protected[i] == 0:
          n_protected_pos += 1
        if x == 1 and protected[i] == 1:
          n_nonprotected_pos += 1
      if (total_sensitive == 0) or (total_nonsensitive == 0):
        return 'NA'
      p_protected_pos = n_protected_pos / float(total_sensitive)
      p_nonprotected_pos = n_nonprotected_pos / float(total_nonsensitive)
      return p_protected_pos - p_nonprotected_pos

    '''
    x = pd.DataFrame(self.x_train)
    datadict = self.x_control_train 
    datadict['class'] = self.y_train
    df = pd.DataFrame.from_dict(datadict)
    print(df)
    hires = df.groupby(['class','Race']).size()
    print(hires)
    '''
    startTime = datetime.now()
    '''
    # Defaults to 1
    if "eta" in list(self.params.keys()):
      eta = self.params["eta"]
    else:
      eta = 1
    '''
    
    x_train_with_sensitive_feature = []

    for i in range(0, len(self.x_train)):
      val = self.x_control_train[self.sensitive_attr][i]
      feature_array = np.append(self.x_train[i], val)
      x_train_with_sensitive_feature.append(feature_array)

    x_train_with_sensitive_feature = np.array(x_train_with_sensitive_feature)

    x_test_with_sensitive_feature = []
    for i in range(0, len(self.x_test)):
      val = self.x_control_test[self.sensitive_attr][i]
      feature_array = np.append(self.x_test[i], val)
      x_test_with_sensitive_feature.append(feature_array)

    x_test_with_sensitive_feature = np.array(x_test_with_sensitive_feature)
    fixed_y_test = []
    for j in self.y_test:
        if j == 1.0:
            fixed_y_test.append(1)
        elif j == -1.0 or j == 0.0:
            fixed_y_test.append(0)
        else:
            print("Invalid class value in y_control_test")

    def getScore(fixed_y_test,y_classified_results, var=1): #var = |1-DI|/acc, 2-acc, 3-|1-DI|
      total_sensitive = 0
      total_nonsensitive = 0
      for x in self.x_control_test[self.sensitive_attr]:
        if x == 0:
          total_sensitive += 1
        elif x == 1:
          total_nonsensitive += 1
      accuracy = accuracy_score(fixed_y_test, y_classified_results)
      if var == 2:
        return accuracy
      elif var == 1:
        return abs(1 - DI_score(y_classified_results, self.x_control_test[self.sensitive_attr], total_sensitive, total_nonsensitive))/accuracy
      else:
        return abs(1-DI_score(y_classified_results, self.x_control_test[self.sensitive_attr], total_sensitive, total_nonsensitive))

    def binMinMax(first, last, var):  #var = |1-DI|/acc, 2-acc, 3-|1-DI|
      if first == last:
        print(first)
        y = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, first, self.x_control_test)
        return y
      elif (last-1) == first:
        firstScore = getScore(fixed_y_test, train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, first, self.x_control_test),var)
        lastScore = getScore(fixed_y_test, train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, last, self.x_control_test),var)
        if firstScore <= lastScore:
          if var == 1 or var == 3:
            return binMinMax(first,first,var)
          else:
            return binMinMax(last,last,var)
        else:
          if var == 1 or var == 3:
            return binMinMax(last,last,var)
          else:
            return binMinMax(first,first,var)
      else:
        mid = (first + last)//2
        firstScore = getScore(fixed_y_test, train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, first, self.x_control_test),var)
        lastScore = getScore(fixed_y_test, train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, last, self.x_control_test),var)
        midY = y_classified_results = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, mid, self.x_control_test)
        midScore = getScore(fixed_y_test, midY, var)

        if midScore <= lastScore:
          if var == 1 or var == 3:
            return binMinMax(first,mid,var)
          else:
            return binMinMax(mid,last,var)
        else:
          if var == 1 or var == 3:
            return binMinMax(mid,last,var)
          else:
            return binMinMax(first,mid,var)

    '''
    minDI = (1000,[])
    maxacc = (1000,[])
    maxDI = (1000,[])
    if self.data=="ricci":
      minDI = (50,[])
      maxacc = (50,[])
      maxDI = (50,[])
    '''

    # PRINT ETAS TO SHOW PLATEAU
    if self.data == 'ricci':
      r = 50
    else:
      r = 1000

    def printScores():
      scoresBoth = []
      scoresAcc = []
      scoresDI = []
      for i in range(r):
        y_classified_results = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, i, self.x_control_test)
        scoresBoth.append(getScore(fixed_y_test,y_classified_results))
        scoresAcc.append(getScore(fixed_y_test,y_classified_results, 2))
        scoresDI.append(getScore(fixed_y_test,y_classified_results,3))
      with open("etas/" + self.data + "-etas-" + self.params['num'] + ".csv",'w') as f:
        f.write('|1-DI|/Acc, Acc, |1-DI|' + '\n')
        for i in range(len(scoresBoth)):
          f.write(str(scoresBoth[i]) + ',' + str(scoresAcc[i]) + ',' + str(scoresDI[i]) +'\n')
      f.close()
      print("SCORES WRITTEN.")
    #############################

    '''
    def binMaxVar(first, last, minDI, maxacc, maxDI, var=1): #var = 1-acc/DI, 2-acc, 3-DI
      if first == last:
        print(first)
        if var == 1:
          return minDI
        elif var == 2:
          return maxacc
        else:
          return maxDI
      else:
        midpoint = (first + last)//2
        y_classified_results = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, midpoint, self.x_control_test)

        midScore = getScore(fixed_y_test, y_classified_results, var)

        firstMid = (first + midpoint)//2
        first_y_classified_results = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, firstMid, self.x_control_test)
        firstMidScore = getScore(fixed_y_test, first_y_classified_results,var)

        secondMid = (firstMid + last)//2
        second_y_classified_results = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, secondMid, self.x_control_test)
        secondMidScore = getScore(fixed_y_test, second_y_classified_results,var)
        
        if firstMidScore <= secondMidScore:
          if var == 1:
            if firstMidScore <= minDI[0]:
              return binMaxVar(first, midpoint, (firstMidScore, first_y_classified_results), maxacc,maxDI)
            else:
              return binMaxVar(first, midpoint, minDI, maxacc, maxDI)
          elif var == 2:
            if firstMidScore <= maxacc[0]:
              return binMaxVar(first, midpoint, minDI, (firstMidScore,first_y_classified_results), maxDI)
            else:
              return binMaxVar(first, midpoint, minDI, maxacc, maxDI)
          else:
            if abs(1-firstMidScore) <= abs(1-maxDI[0]):
              return binMaxVar(first, midpoint, minDI, maxacc, (firstMidScore,first_y_classified_results))
            else:
              return binMaxVar(first, midpoint, minDI, maxacc, maxDI)
          
        else: 
          if var == 1: 
            if secondMidScore <= minDI[0]:
              return binMaxVar(midpoint, last, (secondMidScore,second_y_classified_results), maxacc,maxDI)
            else:
              return binMaxVar(first, midpoint, minDI, maxacc,maxDI)
          elif var == 2:
            if secondMidScore <= maxacc[0]:
              return binMaxVar(midpoint, last, minDI, (secondMidScore, second_y_classified_results),maxDI)
            else:
              return binMaxVar(first, midpoint, minDI, maxacc,maxDI)
          else:
            if abs(1-secondMidScore) <= abs(1-maxDI[0]):
              return binMaxVar(midpoint, last, minDI, maxacc, (secondMidScore, second_y_classified_results))
            else:
              return binMaxVar(first, midpoint, minDI, maxacc,maxDI)

    if self.data == "ricci":
      final = binMaxVar(1,50, minDI, maxacc, maxDI, self.params['var'])
      #final = (0, train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, 1, self.x_control_test))
    else:
      final = binMaxVar(1,1000, minDI, maxacc, maxDI, self.params['var']) 
    predicted = final[1]
    '''

    if self.params['print'] == 1:
      printScores()

    if self.data == "ricci":
      final = binMinMax(1,50,self.params['var'])
    else:
      final = binMinMax(1,1000,self.params['var'])

    # predicted = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, 100, self.x_control_test)

    kam_actual, kam_predicted, kam_protected = fixed_y_test, final, self.x_control_test[self.sensitive_attr]
    kam_time = datetime.now() - startTime

    return kam_actual, kam_predicted, kam_protected, kam_time

def test(data):
  params = {}
  params["eta"] = 1
  algorithm = KamishimaAlgorithm(data, params) 
  print(algorithm.run())

  params["eta"] = 30
  algorithm = KamishimaAlgorithm(data, params)
  print(algorithm.run())

if __name__ == "__main__":
  test("german")
  test("adult")
