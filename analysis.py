import fire
import pandas as pd
from ggplot import *
import sys

# from data.objects.list import DATASETS, get_dataset_names
# from data.objects.ProcessedData import TAGS
# from metrics.list import get_metrics

# def run(dataset = get_dataset_names()):
#     for dataset_obj in DATASETS:
#         if not dataset_obj.get_dataset_name() in dataset:
#             continue

#         all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
#         for sensitive in all_sensitive_attributes:

#             # Write summary files and graphs per dataset
#             for tag in TAGS:
#                 write_summary_file(dataset_obj.get_results_filename(sensitive, tag),
#                                    dataset_obj.get_analysis_filename(sensitive, tag), dataset_obj)
#                 make_graph(dataset_obj.get_dataset_name() + " dataset, " + sensitive + " - " + tag,
#                            dataset_obj.get_analysis_filename(sensitive, tag))

# def write_summary_file(infile, outfile, dataset):
#     outf = open(outfile, 'w')
#     outf.write(summary_file_header(dataset))
#     df = pd.read_csv(infile)
#     algorithms = df.algorithm.unique()
#     for alg in algorithms:
#         line = alg
#         alg_rows = df.loc[df['algorithm'] == alg]
#         for metric in get_metrics(dataset):
#             metric_vals = alg_rows[metric.get_name()].values.tolist()
#             line += ',' + str(statistics.mean(metric_vals)) + ',' + \
#                     str(statistics.stdev(metric_vals))
#         outf.write(line + '\n')
#     outf.close()
#     print("Wrote summary file:" + outfile)

# def make_graph(graph_title, summary_file):
#     df = pd.read_csv(summary_file)
#     sns.lmplot(x='DIavgall', y='accuracy', data=df, fit_reg=False, hue='algorithm',
#                scatter_kws={"s": 100}, legend=True, legend_out=False)
#     ax = plt.gca()
#     ax.set_title(graph_title)
#     plt.axvline(x=1.0)
#     plt.show()

# def summary_file_header(dataset):
#     line = "algorithm," + ",".join([x.get_name() + ",stdev" for x in get_metrics(dataset)]) + "\n"
#     return line

# def main():
#     fire.Fire(run)

##############################################################################

import pathlib


def main():
    f = pd.read_csv(sys.argv[1])
    o = pathlib.Path(sys.argv[1]).parts[-1].split('.')[0]
    measures = list(f.columns.values)

    for i, m1 in enumerate(measures[2:]):
        for j, m2 in enumerate(measures[2:]):
            pathlib.Path("results/analysis/%s" % o).mkdir(parents=True, exist_ok=True)
            # scale = scale_color_brewer(type='qual', palette=1)
            # d3.schemeCategory20
            # ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
            #  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
            #  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
            scale = scale_color_manual(values=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
                                               "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
                                               "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
            p = (ggplot(f, aes(x=m1, y=m2, colour='algorithm')) +
                 geom_point(size=50) + scale)
            print(m1, m2)
            p.save('results/analysis/%s/%s-%s.png' % (o, m1, m2),
                   width=8,
                   height=6)

if __name__ == '__main__':
    main()
