import fire
import pandas as pd
import pathlib
import sys

from ggplot import *

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import TAGS

# The names of the metrics to generate graphs for.
METRIC_NAMES = ['accuracy', 'DIbinary', 'DIavgall', 'CV']

def run(dataset = get_dataset_names(), metrics = METRIC_NAMES):
    """
    If 'all' is given as the value for metrics, all possible combinations of metric graphs are
    generated.
    """
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nGenerating graphs for dataset:" + dataset_obj.get_dataset_name())
        for sensitive in dataset_obj.get_sensitive_attributes_with_joint():
            for tag in TAGS:
                print("    type:" + tag)
                filename = dataset_obj.get_results_filename(sensitive, tag)
                generate_graphs(filename, metrics)

def generate_graphs(filename, measures):
    f = pd.read_csv(filename)
    o = pathlib.Path(filename).parts[-1].split('.')[0]

    if measures == 'all':
        measures = list(f.columns.values)[2:]

    for i, m1 in enumerate(measures):
        col1 = f[m1]

        if len(col1) == 0:
            print("Skipping file containing no data:" + filename)
            return

        if col1[0] == 'None':
            print("Skipping missing column %s" % m1)
            continue
        for j, m2 in enumerate(measures):
            col2 = f[m2]
            if col2[0] == 'None':
                print("Skipping missing column %s" % m2)
                continue
            pathlib.Path("results/analysis/%s" % o).mkdir(parents=True, exist_ok=True)
            # scale = scale_color_brewer(type='qual', palette=1)
            # d3.schemeCategory20
            # ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
            #  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
            #  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
            scale = scale_color_manual(
                        values=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                                "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
                                "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
                                "#17becf", "#9edae5"])
            p = (ggplot(f, aes(x=m1, y=m2, colour='algorithm')) +
                 geom_point(size=50) + ggtitle(o) + scale)
            print(m1, m2)
            p.save('results/analysis/%s/%s-%s.png' % (o, m1, m2),
                   width=20,
                   height=6)

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
