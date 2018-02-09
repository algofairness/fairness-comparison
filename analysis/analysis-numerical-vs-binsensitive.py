import fire
import pandas as pd
from ggplot import *
import sys

##############################################################################

import pathlib

def main():
    o = pathlib.Path(sys.argv[1]).parts[-1].split('.')[0]
    f1 = pd.read_csv(sys.argv[1] + '_numerical-binsensitive.csv')
    f2 = pd.read_csv(sys.argv[1] + '_numerical.csv')
    scale = scale_color_manual(values=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
                                       "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
                                       "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
    f1["algo-run-id"] = list('%s-%s' % (l, r) for (l, r) in zip(f1['algorithm'], f1['run-id']))
    f2["algo-run-id"] = list('%s-%s' % (l, r) for (l, r) in zip(f2['algorithm'], f2['run-id']))
    
    j = f1.merge(f2, on='algo-run-id', suffixes =('-numerical-binsensitive', '-numerical'))
    j["algorithm"] = j["algorithm-numerical"]

    attributes = f1.columns.values[3:]
    pathlib.Path("results/analysis/%s-comparison" % o).mkdir(parents=True, exist_ok=True)
            
    for attribute in attributes:
        x = '%s-numerical-binsensitive' % attribute
        y = '%s-numerical' % attribute
        if x not in j.columns.values or y not in j.columns.values:
            print("Skipping attribute %s" % attribute, file=sys.stderr)
            continue
        print(attribute)
        p = ggplot(j, aes(x=x, y=y)) + facet_wrap('algorithm') + geom_point(size=50)
        p.save('results/analysis/%s-comparison/%s-numerical-binsensitive.png' % (o, attribute), width=8, height=6)

    # o = pathlib.Path(sys.argv[1]).parts[-1].split('.')[0]
    # measures = list(f.columns.values)

    # for i, m1 in enumerate(measures[2:]):
    #     col1 = f[m1]
    #     if col1[0] == 'None':
    #         print("Skipping missing column %s" % m1)
    #         continue
    #     for j, m2 in enumerate(measures[2:]):
    #         col2 = f[m2]
    #         if col2[0] == 'None':
    #             print("Skipping missing column %s" % m2)
    #             continue
    #         pathlib.Path("results/analysis/%s" % o).mkdir(parents=True, exist_ok=True)
    #         # scale = scale_color_brewer(type='qual', palette=1)
    #         # d3.schemeCategory20
    #         # ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    #         #  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    #         #  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
    #         scale = scale_color_manual(values=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    #                                            "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    #                                            "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
    #         p = (ggplot(f, aes(x=m1, y=m2, colour='algorithm')) +
    #              geom_point(size=50) + scale)
    #         print(m1, m2)
    #         p.save('results/analysis/%s/%s-%s.png' % (o, m1, m2),
    #                width=8,
    #                height=6)

if __name__ == '__main__':
    main()
