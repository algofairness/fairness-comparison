
## Data sources and citations

### German Credit Data
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/

### Adult Income Data
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

### Ricci Data
Source: https://ww2.amstat.org/publications/jse/v18n3/RicciData.csv 

Citation: Miao, Weiwen. "Did the results of promotion exams have a disparate impact on minorities? Using statistical evidence in Ricci v. DeStefano." Journal of Statistics Education 18.3 (2010): 1-26.
http://ww2.amstat.org/publications/jse/v18n3/miao.pdf

### Retailer Data
Source: https://www.dropbox.com/sh/r1q4ha5c0corzmt/AACYL2SMB_nxG6LNbXeGjXKAa?dl=0&preview=cleaned-apps-public-csv.zip


## Adding a data set

To add a data set, you need to:
1. Choose a single word lower case *name* to identify your data set.
2. Put the raw data set in the raw/ directory at *name*.csv.  Add any data info at *name*.txt.
3. Create a class *Name*.py that extends Data.py and implements all the required methods.  Add it to datasets/  A Stub.py file is provided that you can copy and edit.
4. Add your dataset object to the list at datasets/list.py


## Generating the preprocessed versions of the data

All preprocessed versions of the data should be committed to the preprocessed directory.
To regenerate them, run:
> python preprocess.py
