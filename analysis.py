import fire
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data.objects.list import DATASETS, get_dataset_names
from metrics.list import METRICS

def run(dataset = get_dataset_names()):
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes()
        if len(all_sensitive_attributes) > 1:
            # add the joint sensitive attribute (e.g., race-sex) to the list
            all_sensitive_attributes += [ dataset_obj.get_combined_sensitive_attr_name() ]

        for sensitive in all_sensitive_attributes:

            # Write summary files per dataset
            write_summary_file(dataset_obj.get_results_numerical_binsensitive_filename(sensitive),
                               dataset_obj.get_analysis_numerical_binsensitive_filename(sensitive))
            write_summary_file(dataset_obj.get_results_numerical_filename(sensitive),
                               dataset_obj.get_analysis_numerical_filename(sensitive))
            write_summary_file(dataset_obj.get_results_filename(sensitive),
                               dataset_obj.get_analysis_filename(sensitive))

            # Write graphs per dataset
            make_graph(dataset_obj.get_dataset_name() + " dataset, " + sensitive + \
                           " - numerical and binary sensitive",
                       dataset_obj.get_analysis_numerical_binsensitive_filename(sensitive))
            make_graph(dataset_obj.get_dataset_name() + " dataset, " + sensitive + \
                           " - numerical",
                       dataset_obj.get_analysis_numerical_filename(sensitive))
            make_graph(dataset_obj.get_dataset_name() + " dataset, " + sensitive + \
                           "- numerical and categorical",
                       dataset_obj.get_analysis_filename(sensitive))

def write_summary_file(infile, outfile):
    outf = open(outfile, 'w')
    outf.write(summary_file_header())
    df = pd.read_csv(infile)
    algorithms = df.algorithm.unique()
    for alg in algorithms:
        line = alg
        alg_rows = df.loc[df['algorithm'] == alg]
        for metric in METRICS:
            metric_vals = alg_rows[metric.get_name()].values.tolist()
            line += ',' + str(statistics.mean(metric_vals)) + ',' + \
                    str(statistics.stdev(metric_vals))
        outf.write(line + '\n')
    outf.close()
    print("Wrote summary file:" + outfile)

def make_graph(graph_title, summary_file):
    df = pd.read_csv(summary_file)
    sns.lmplot(x='DisparateImpact', y='accuracy', data=df, fit_reg=False, hue='algorithm',
               scatter_kws={"s": 100}, legend=True, legend_out=False)
    ax = plt.gca()
    ax.set_title(graph_title)
    plt.axvline(x=1.0)
    plt.show()

def summary_file_header():
    line = "algorithm," + ",".join([x.get_name() + ",stdev" for x in METRICS]) + "\n"
    return line

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
