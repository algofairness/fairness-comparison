import fire
import math
import pandas as pd

from fairness.data.objects.Adult import Adult
from fairness.data.objects.ProcessedData import TAGS

def run(algname, data = Adult(), measure = 'accuracy'):
    for filename in make_filenames(algname, measure, data):
        outfile = filename + '.correctedbest.csv'
        out = open(outfile, 'w')
        print(filename)
        f = pd.read_csv(filename)
        try:
            params = f['params'][0]
        except:
            print("Skipping empty file:" + filename)
            continue
        else:
            run_id = 0
            run_id_best_line = 0
            start_param = f['params'][0]
            for i, (alg, meas) in enumerate(zip(f['algorithm'], f[measure])):
                params = f['params'][i]
                if params == start_param:
                    if run_id != 0:
                        print(run_id, run_id_best_val)
                        line_str = ','.join([str(x) for x in
                                             f.ix[run_id_best_line].values.tolist()]) + '\n'
                        out.write(line_str)
                    run_id += 1
                    run_id_best_line = i
                    run_id_best_val = meas
                if is_better_than(meas, run_id_best_val, measure):
                    run_id_best_line = i
                    run_id_best_val = meas
            print(run_id, run_id_best_val)
            line_str = ','.join([str(x) for x in
                                 f.ix[run_id_best_line].values.tolist()]) + '\n'
            out.write(line_str)
        out.close()
        print("Corrected best per split written to:" + outfile)

def is_better_than(val1, val2, measure):
    if measure == 'accuracy':
        return val1 >= val2
    if 'DI' in measure:
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2
    print("ERROR: Unkown measure:" + measure)
    return

def make_filenames(algname, measure, dataobj):
    names = []
    for sensitive in dataobj.get_sensitive_attributes_with_joint():
        for tag in TAGS:
            print("    type:" + tag)
            dataname = dataobj.get_dataset_name()
            filename = 'results/' + algname + '-' + measure + '_' + dataname + '_' + sensitive + \
                       '_' + tag + '.csv'
            names.append(filename)
    return names

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
