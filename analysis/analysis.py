import fire
import pandas as pd
from ggplot import *
import sys

##############################################################################

import pathlib

def main():
    f = pd.read_csv(sys.argv[1])
    o = pathlib.Path(sys.argv[1]).parts[-1].split('.')[0]

    if len(sys.argv) > 2:
        measures = sys.argv[2:]
    else:
        measures = list(f.columns.values)[2:]

    for i, m1 in enumerate(measures):
        col1 = f[m1]
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
            scale = scale_color_manual(values=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
                                               "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
                                               "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
            p = (ggplot(f, aes(x=m1, y=m2, colour='algorithm')) +
                 geom_point(size=50) + ggtitle(o) + scale)
            print(m1, m2)
            p.save('results/analysis/%s/%s-%s.png' % (o, m1, m2),
                   width=8,
                   height=6)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage:")
        print("  %s RESULT_FILE [measure1 measure2 measure...]" % sys.argv[0])
        print("  (If no measures are specified, all pairwise measures are generated: this is potentially large)")
        exit(1)
    main()
