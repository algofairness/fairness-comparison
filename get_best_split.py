import fire
import pandas as pd

from data.objects.Adult import Adult
from data.objects.ProcessedData import TAGS

def run(algname, data = Adult(), measure = 'accuracy'):
    for filename in make_filenames(algname, measure, data):
        outfile = filename + '.correctedbest'
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
            for i, (alg, meas) in enumerate(zip(f['algorithm'], f[measure])):
                params = f['params'][i]
                if params == 'lambda=0.0':
                    if run_id != 0:
                        line_str = ','.join([str(x) for x in
                                             f.ix[run_id_best_line].values.tolist()]) + '\n'
                        out.write(line_str)
                    run_id += 1
                    run_id_best_line = i
                    run_id_best_val = meas
                if meas >= run_id_best_val:
                    run_id_best_line = i
                    run_id_best_val = meas
            line_str = ','.join([str(x) for x in
                                 f.ix[run_id_best_line].values.tolist()]) + '\n'
            out.write(line_str)
        out.close()
        print("Corrected best per split written to:" + outfile)

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
