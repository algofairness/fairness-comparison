from algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from algorithms.baseline.SVM import SVM
from algorithms.baseline.DecisionTree import DecisionTree
from algorithms.baseline.GaussianNB import GaussianNB
from algorithms.baseline.LogisticRegression import LogisticRegression
from algorithms.ParamGridSearch import ParamGridSearch
from algorithms.Ben.SDBAlgorithm import SDBAlgorithm

from metrics.DisparateImpact import DisparateImpact
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC

ALGORITHMS = [ SVM(), GaussianNB(), LogisticRegression(), DecisionTree(), # baseline
              SDBAlgorithm(SVM()),                                       #SDBAlgorithm                           
              KamishimaAlgorithm(),                                          # Kamishima
               ParamGridSearch(KamishimaAlgorithm(), Accuracy()),             # Kamishima params
               ParamGridSearch(KamishimaAlgorithm(), DisparateImpact()),
               ParamGridSearch(KamishimaAlgorithm(), EqOppo_fp_diff()),
               ParamGridSearch(KamishimaAlgorithm(), EqOppo_fn_diff()),
               ParamGridSearch(KamishimaAlgorithm(), EqOppo_fp_ratio()),
               ParamGridSearch(KamishimaAlgorithm(), EqOppo_fp_ratio()), 
               FeldmanAlgorithm(SVM()), FeldmanAlgorithm(GaussianNB()),       # Feldman
               FeldmanAlgorithm(LogisticRegression()), FeldmanAlgorithm(DecisionTree()),
               ParamGridSearch(FeldmanAlgorithm(SVM()), DisparateImpact()),   # Feldman params
               ParamGridSearch(FeldmanAlgorithm(SVM()), Accuracy()),
               ParamGridSearch(FeldmanAlgorithm(SVM()), EqOppo_fp_diff()),
               ParamGridSearch(FeldmanAlgorithm(SVM()), EqOppo_fn_diff()),
               ParamGridSearch(FeldmanAlgorithm(SVM()), EqOppo_fp_ratio()),
               ParamGridSearch(FeldmanAlgorithm(SVM()), EqOppo_fn_ratio()),
               ParamGridSearch(FeldmanAlgorithm(GaussianNB()), DisparateImpact()),
               ParamGridSearch(FeldmanAlgorithm(GaussianNB()), Accuracy()),
               ParamGridSearch(FeldmanAlgorithm(GaussianNB()), EqOppo_fp_diff()),
               ParamGridSearch(FeldmanAlgorithm(GaussianNB()), EqOppo_fn_diff()),
               ParamGridSearch(FeldmanAlgorithm(GaussianNB()), EqOppo_fp_ratio()),
               ParamGridSearch(FeldmanAlgorithm(GaussianNB()), EqOppo_fn_ratio())
             ]
