import sys
import pandas as pd

ATTRS_TO_SAVE = [
  "algorithm",
  "params",
  "run-id",

  "DIbinary",
  "DIavgall",
  "CV",
  "comparative-sensitive-TPR",         # e.g. name in data: race-TPRDiff
  "accuracy",
  "0-accuracy",
  "1-accuracy",
  "sensitive-accuracy",                # e.g. name in data: race-accuracy
  "TNR",
  "sensitive-TNR",                     # e.g. name in data: race-TNR
  "comparative-sensitive-accuracy",    # e.g. name in data: race-accuracyDiff
  "BCR",
  "sensitive-calibration+",
  "sensitive-calibration-",
  "TPR",
  "sensitive-TPR",
  "comparative-sensitive-TNR",
  "MCC",
  "comparative-sensitive-calibration+",
  "comparative-sensitive-calibration-"
]

def combine_files(files, outfile):
    combined = pd.DataFrame()
    for filename in files:
        print("Processing:" + filename)
        dataframe = pd.read_csv(filename)
        numrows, nummetrics = dataframe.shape
        print("    " + str(numrows) + " items")
        dataname, sensitive = get_sensitive_from_filename(filename)
        metrics = make_metrics_list(sensitive)
        dataframe = dataframe[metrics]
        dataframe.insert(0, 'attribute', sensitive)
        dataframe.insert(0, 'name', dataname)
        dataframe = dataframe[1:]  # remove attribute specific header
        # replace with generalized header
        dataframe.columns = ["name", "attribute"] + ATTRS_TO_SAVE
        combined = combined.append(dataframe)
    print(str(combined.shape[0]) + " total items, " + str(combined.shape[1]) + " metrics")
    combined.to_csv(outfile)
    print("CSV written to:" + outfile)

def get_sensitive_from_filename(filename):
    dataset, sensitive, tag = filename.split("/")[-1].split("_")
    return dataset, sensitive

def make_metrics_list(sensitive_attr):
    revised_metrics = []
    for metric in ATTRS_TO_SAVE:
        if "sensitive" in metric:
            revised = metric.replace("sensitive", sensitive_attr)
        else:
            revised = metric
        if "comparative-" in revised:
            revised = revised.replace("comparative-", "")
            revised = revised + "Diff"
        revised_metrics.append(revised)
    return revised_metrics

def main():
    if len(sys.argv) < 2:
        print("Given a list of files in the results format, this script combines them into a file")
        print("in the correlation vis format.")
        print("Usage: python3 combine_results_files.py <files list> outfile")
        exit(-1)
    combine_files(sys.argv[1:-1], sys.argv[-1])

if __name__ == '__main__':
    main()
