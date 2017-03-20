Thu Mar 16 15:06:25 EDT 2017
learning script: train_lr.py
test script:  + predict_lr.py
result file: 00RESULT/adultd@method=LR-reg=1@t.result
model: 00MODEL/adultd@method=LR-reg=1@0l.model
adultd@method=LR-reg=1@0
learning script: train_lr.py
test script:  + predict_lr.py
result file: 00RESULT/adultd@method=LR-reg=1@t.result
model: 00MODEL/adultd@method=LR-reg=1@1l.model
adultd@method=LR-reg=1@1
learning script: train_lr.py
test script:  + predict_lr.py
result file: 00RESULT/adultd@method=LR-reg=1@t.result
model: 00MODEL/adultd@method=LR-reg=1@2l.model
adultd@method=LR-reg=1@2
learning script: train_lr.py
test script:  + predict_lr.py
result file: 00RESULT/adultd@method=LR-reg=1@t.result
model: 00MODEL/adultd@method=LR-reg=1@3l.model
adultd@method=LR-reg=1@3
learning script: train_lr.py
test script:  + predict_lr.py
result file: 00RESULT/adultd@method=LR-reg=1@t.result
model: 00MODEL/adultd@method=LR-reg=1@4l.model
adultd@method=LR-reg=1@4
### adultd@method=LR-reg=1
learning script: train_lr.py --ns
test script:  + predict_lr.py --ns
result file: 00RESULT/adultd@method=LRns-reg=1@t.result
model: 00MODEL/adultd@method=LRns-reg=1@0l.model
adultd@method=LRns-reg=1@0
learning script: train_lr.py --ns
test script:  + predict_lr.py --ns
result file: 00RESULT/adultd@method=LRns-reg=1@t.result
model: 00MODEL/adultd@method=LRns-reg=1@1l.model
adultd@method=LRns-reg=1@1
learning script: train_lr.py --ns
test script:  + predict_lr.py --ns
result file: 00RESULT/adultd@method=LRns-reg=1@t.result
model: 00MODEL/adultd@method=LRns-reg=1@2l.model
adultd@method=LRns-reg=1@2
learning script: train_lr.py --ns
test script:  + predict_lr.py --ns
result file: 00RESULT/adultd@method=LRns-reg=1@t.result
model: 00MODEL/adultd@method=LRns-reg=1@3l.model
adultd@method=LRns-reg=1@3
learning script: train_lr.py --ns
test script:  + predict_lr.py --ns
result file: 00RESULT/adultd@method=LRns-reg=1@t.result
model: 00MODEL/adultd@method=LRns-reg=1@4l.model
adultd@method=LRns-reg=1@4
### adultd@method=LRns-reg=1
adultd@method=NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@0
adultd@method=NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@1
adultd@method=NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@2
adultd@method=NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@3
adultd@method=NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@4
### adultd@method=NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8
adultd@method=NBns-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@0
adultd@method=NBns-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@1
adultd@method=NBns-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@2
adultd@method=NBns-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@3
adultd@method=NBns-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@4
### adultd@method=NBns-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8
Traceback (most recent call last):
  File "train_cv2nb.py", line 254, in <module>
    main(opt)
  File "train_cv2nb.py", line 128, in main
    clr.fit(X, y, ns)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 155, in fit
    numpos, disc = self._get_stats(X, y)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 170, in _get_stats
    py = self.predict(X)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 62, in predict
    log_proba = self._predict_log_proba_upto_const(np.array(X))
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 200, in _predict_log_proba_upto_const
    XX[s == si, :]) + \
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 637, in _predict_composite_log_proba_upto_const
    self._predict_multinomial_log_proba_upto_const(X[i, self.mfeatures])
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 502, in _predict_multinomial_log_proba_upto_const
    log_proba = np.sum([p(i) for i in f], axis=0)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 501, in <lambda>
    - np.log(np.sum(self.pf_[i], axis=1))
KeyboardInterrupt
Traceback (most recent call last):
  File "predict_nb.py", line 278, in <module>
    main(opt)
  File "predict_nb.py", line 109, in main
    clr = pickle.load(opt.model)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 1384, in load
    return Unpickler(file).load()
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 864, in load
    dispatch[key](self)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 886, in load_eof
    raise EOFError
EOFError
adultd@method=CV2NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@0
Traceback (most recent call last):
  File "train_cv2nb.py", line 254, in <module>
    main(opt)
  File "train_cv2nb.py", line 128, in main
    clr.fit(X, y, ns)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 136, in fit
    numpos, disc = self._get_stats(X, y)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 170, in _get_stats
    py = self.predict(X)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 62, in predict
    log_proba = self._predict_log_proba_upto_const(np.array(X))
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 200, in _predict_log_proba_upto_const
    XX[s == si, :]) + \
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 637, in _predict_composite_log_proba_upto_const
    self._predict_multinomial_log_proba_upto_const(X[i, self.mfeatures])
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 502, in _predict_multinomial_log_proba_upto_const
    log_proba = np.sum([p(i) for i in f], axis=0)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 501, in <lambda>
    - np.log(np.sum(self.pf_[i], axis=1))
KeyboardInterrupt
Traceback (most recent call last):
  File "predict_nb.py", line 278, in <module>
    main(opt)
  File "predict_nb.py", line 109, in main
    clr = pickle.load(opt.model)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 1384, in load
    return Unpickler(file).load()
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 864, in load
    dispatch[key](self)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 886, in load_eof
    raise EOFError
EOFError
adultd@method=CV2NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@1
Traceback (most recent call last):
  File "train_cv2nb.py", line 254, in <module>
    main(opt)
  File "train_cv2nb.py", line 128, in main
    clr.fit(X, y, ns)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 136, in fit
    numpos, disc = self._get_stats(X, y)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 170, in _get_stats
    py = self.predict(X)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 62, in predict
    log_proba = self._predict_log_proba_upto_const(np.array(X))
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 200, in _predict_log_proba_upto_const
    XX[s == si, :]) + \
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 637, in _predict_composite_log_proba_upto_const
    self._predict_multinomial_log_proba_upto_const(X[i, self.mfeatures])
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 502, in _predict_multinomial_log_proba_upto_const
    log_proba = np.sum([p(i) for i in f], axis=0)
KeyboardInterrupt
Traceback (most recent call last):
  File "predict_nb.py", line 70, in <module>
    import numpy as np
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/__init__.py", line 167, in <module>
    from . import random
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/random/__init__.py", line 99, in <module>
    from .mtrand import *
KeyboardInterrupt
adultd@method=CV2NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@2
Traceback (most recent call last):
  File "train_cv2nb.py", line 65, in <module>
    import numpy as np
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/__init__.py", line 146, in <module>
    from . import add_newdocs
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
    from numpy.lib import add_newdoc
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
    from .type_check import *
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
    import numpy.core.numeric as _nx
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/core/__init__.py", line 25, in <module>
    from . import numeric
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/core/numeric.py", line 21, in <module>
    import cPickle as pickle
  File "<string>", line 1, in <module>
KeyboardInterrupt
adultd@method=CV2NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@3
Traceback (most recent call last):
  File "train_cv2nb.py", line 254, in <module>
    main(opt)
  File "train_cv2nb.py", line 128, in main
    clr.fit(X, y, ns)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 155, in fit
    numpos, disc = self._get_stats(X, y)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 170, in _get_stats
    py = self.predict(X)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 62, in predict
    log_proba = self._predict_log_proba_upto_const(np.array(X))
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/cv2nb.py", line 200, in _predict_log_proba_upto_const
    XX[s == si, :]) + \
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 637, in _predict_composite_log_proba_upto_const
    self._predict_multinomial_log_proba_upto_const(X[i, self.mfeatures])
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 502, in _predict_multinomial_log_proba_upto_const
    log_proba = np.sum([p(i) for i in f], axis=0)
  File "/Users/ephamilton/Desktop/cs_thesis/2012ecmlpkdd/fadm/nb/_nb.py", line 501, in <lambda>
    - np.log(np.sum(self.pf_[i], axis=1))
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/core/fromnumeric.py", line 1848, in sum
    out=out, **kwargs)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/site-packages/numpy/core/_methods.py", line 32, in _sum
    return umr_sum(a, axis, dtype, out, keepdims)
KeyboardInterrupt
Traceback (most recent call last):
  File "predict_nb.py", line 278, in <module>
    main(opt)
  File "predict_nb.py", line 109, in main
    clr = pickle.load(opt.model)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 1384, in load
    return Unpickler(file).load()
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 864, in load
    dispatch[key](self)
  File "/Users/ephamilton/miniconda2/envs/kamashima/lib/python2.7/pickle.py", line 886, in load_eof
    raise EOFError
EOFError
adultd@method=CV2NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8@4
### adultd@method=CV2NB-beta=1.0-nfv=4:7:4:16:4:7:14:6:5:2:2:3:8
