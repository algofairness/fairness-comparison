import fire
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import TAGS
from metrics.list import get_metrics

def run(dataset = get_dataset_names()):
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        for sensitive in all_sensitive_attributes:

            # Write summary files and graphs per dataset
            for tag in TAGS:
                write_summary_file(dataset_obj.get_results_filename(sensitive, tag),
                                   dataset_obj.get_analysis_filename(sensitive, tag), dataset_obj)
                make_graph(dataset_obj.get_dataset_name() + " dataset, " + sensitive + " - " + tag,
                           dataset_obj.get_analysis_filename(sensitive, tag))

def write_summary_file(infile, outfile, dataset):
    outf = open(outfile, 'w')
    outf.write(summary_file_header(dataset))
    df = pd.read_csv(infile)
    algorithms = df.algorithm.unique()
    for alg in algorithms:
        line = alg
        alg_rows = df.loc[df['algorithm'] == alg]
        for metric in get_metrics(dataset):
            metric_vals = alg_rows[metric.get_name()].values.tolist()
            line += ',' + str(statistics.mean(metric_vals)) + ',' + \
                    str(statistics.stdev(metric_vals))
        outf.write(line + '\n')
    outf.close()
    print("Wrote summary file:" + outfile)

def make_graph(graph_title, summary_file):
    df = pd.read_csv(summary_file)
    sns.lmplot(x='DIavgall', y='accuracy', data=df, fit_reg=False, hue='algorithm',
               scatter_kws={"s": 100}, legend=True, legend_out=False)
    ax = plt.gca()
    ax.set_title(graph_title)
    plt.axvline(x=1.0)
    plt.show()

def summary_file_header(dataset):
    line = "algorithm," + ",".join([x.get_name() + ",stdev" for x in get_metrics(dataset)]) + "\n"
    return line

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
