### task list
### included from gxp-all-task.sh or test-all-task.sh

##############################

script=learn-cv-lr.sh

for reg in 1; do

# logistic regression with a sensitive feature
method=LR
lscript=train_lr.py
tscript=predict_lr.py
go "reg=${reg}"

# logistic regression without a sensitive feature
method=LRns
lscript="train_lr.py --ns"
tscript="predict_lr.py --ns"
go "reg=${reg}"

done

##############################

script=learn-cv-nb.sh
beta=1.0

# naive Bayes with a sensitive feature
method=NB
lscript=train_nb.py
tscript=predict_nb.py
go "beta=${beta} nfv=${nfv}"

# naive Bayes without a sensitive feature
method=NBns
lscript="train_nb.py --ns"
tscript="predict_nb.py --ns"
go "beta=${beta} nfv=${nfv}"

# Calders-Verwer 2-naive Bayes
method=CV2NB
lscript=train_cv2nb.py
tscript=predict_nb.py
go "beta=${beta} nfv=${nfv}"

##############################

script=learn-cv-lr.sh

ntry=1
for reg in 1; do
for ltype in 4; do
for itype in 3; do
for eta in 0.0 5.0 10.0 15.0 20.0 30.0; do

# prejudice remover
method=PR${ltype}
lscript=train_pr.py
tscript=predict_lr.py
go "reg=${reg} eta=${eta} ltype=${ltype} itype=${itype} try=${ntry}"

done
done
done
done

##############################
