
# Algorithms included

## Kamishima et al.

### Code source:
https://github.com/tkamishima/kamfadm/releases/tag/2012ecmlpkdd

### Paper to cite: 
T. Kamishima, S. Akaho, H. Asoh, and J. Sakuma "Fairness-Aware Classifier with Prejudice Remover Regularizer" Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD), Part II, pp.35-50 (2012) 

### More information:
http://www.kamishima.net/faclass/

# Adding a new algorithm

1. Make a new directory named after the first author of the relevant paper.
2. In the new directory create a file named <FirstAuthor>Algorithm.py that extends Algorithm.py and implements its run method.  Read through the other methods and implement any necessary for your algorithm.  Be careful to return predictions from the run method that are of the same type as the class values in the given data, or metric comparisons in these benchmarks may fail.
3. Add any additional needed code in that directory or a subdirectory.
4. Add the algorithm to list.py.  Be sure to also add the ParamGridSearch version(s) of your algorithm if your algorithm has a parameter that can be used for tuning.
5. Add citation information to this README.
