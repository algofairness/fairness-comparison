from algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from algorithms.baseline.SVM import SVM
from algorithms.baseline.GaussianNB import GaussianNB
from algorithms.baseline.LogisticRegression import LogisticRegression

ALGORITHMS = [ SVM(), GaussianNB(), LogisticRegression(), 
               FeldmanAlgorithm(SVM()), FeldmanAlgorithm(GaussianNB()), 
               FeldmanAlgorithm(LogisticRegression()) ]
# KamishimaAlgorithm() ]     # 'feldman', 'calder', 'kamishima', 'zafar', 'gen']
