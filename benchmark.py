import fire
import statistics

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS
from metrics.list import METRICS, FAIRNESS_METRICS

NUM_TRIALS_DEFAULT = 10
RESULT_DIR = "results/"

def run(num_trials = NUM_TRIALS_DEFAULT, dataset_names = get_dataset_names()):
    print("WARNING: be sure that you have run `python3 preprocess.py` before running this script.")
    metrics_line, metrics_list = get_metrics_name_list()

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_names:
            continue

        print("\nEvaluating dataset:" + dataset.get_dataset_name())

        processed_dataset = ProcessedData(dataset)
        processed_splits, numerical_splits = processed_dataset.create_train_test_splits(num_trials)

        all_sensitive_attributes = dataset.get_sensitive_attributes() 
        if len(all_sensitive_attributes) > 1:
            # all the joint sensitive attribute (e.g., race-sex) to the list
            all_sensitive_attributes += [ processed_dataset.get_combined_sensitive_attr_name() ]

        for sensitive in all_sensitive_attributes:

            print("Sensitive attribute:" + sensitive)

            filename_num = RESULT_DIR + dataset.get_dataset_name() + '_' + sensitive + '_numerical.csv' 
            f_num = open(filename_num, 'w')
            f_num.write(metrics_line + '\n')
 
            filename_all = RESULT_DIR + dataset.get_dataset_name() + '_' + sensitive + '_all.csv' 
            f_all = open(filename_all, 'w')
            f_all.write(metrics_line + '\n')


            for algorithm in ALGORITHMS:
                print("    Algorithm:" + algorithm.get_name())
                line_num = algorithm.get_name()
                line_all = algorithm.get_name()
                 
                numeric_results = {}
                alldata_results = {}
                for i in range(0, num_trials):
            
                    if not algorithm.numerical_data_only():
                        train, test = processed_splits[i]
                        run_eval_alg(algorithm, train, test, dataset, all_sensitive_attributes, 
                                     sensitive, alldata_results)
            
                    train, test = numerical_splits[i]
                    run_eval_alg(algorithm, train, test, dataset, all_sensitive_attributes, sensitive,
                                 numeric_results)
            
                for metric_name in metrics_list:
                    if not algorithm.numerical_data_only():
                        avg = statistics.mean(alldata_results[metric_name])
                        stdev = statistics.stdev(alldata_results[metric_name])
                        line_all += ', ' + str(avg) + ', ' + str(stdev)
                    
                    avg = statistics.mean(numeric_results[metric_name])
                    stdev = statistics.stdev(numeric_results[metric_name])
                    line_num += ', ' + str(avg) + ', ' + str(stdev)
            
                f_num.write(line_num + '\n')                
            
                if not algorithm.numerical_data_only():
                    f_all.write(line_all + '\n')

        print("Results written to file(s) in: " + RESULT_DIR)

        f_num.close()
        f_all.close()

def run_eval_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive, 
                 results_dict):
    actual, predicted, sensitive = run_alg(algorithm, train, test, dataset, all_sensitive_attributes,
                                           single_sensitive)

    for metric in METRICS:
        result = metric.calc(actual, predicted) 
        name = metric.get_name()
        if not name in results_dict:
            results_dict[name] = []
        results_dict[name].append(result)

    privileged_vals = dataset.get_privileged_class_names()
    positive_val = dataset.get_positive_class_val()
    for fairness_metric in FAIRNESS_METRICS:
        result = fairness_metric.calc(actual, predicted, sensitive, privileged_vals, positive_val) 
        name = fairness_metric.get_name()
        if not name in results_dict:
            results_dict[name] = []
        results_dict[name].append(result)
 
def run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive):
    class_attr = dataset.get_class_attribute()
    params = {}  ## TODO: algorithm specific parameters still need to be handled

    # get the actual classifications and sensitive attributes
    actual = test[class_attr]
    for attr in all_sensitive_attributes:
       ## TODO: actually deal with multiple protected attributes
       sensitive = test[attr].values.tolist()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done. 
    predictions = algorithm.run(train, test, class_attr, all_sensitive_attributes, single_sensitive, 
                                params)

    return actual, predictions, sensitive

def get_metrics_name_list():
    name_list = []
    name_line = ''
    for metric in METRICS:
        name_list.append(metric.get_name())
        name_line += ', ' + metric.get_name() + ', ' + metric.get_name() + ' stdev'
    for metric in FAIRNESS_METRICS:
        name_list.append(metric.get_name())
        name_line += ', ' + metric.get_name() + ', ' + metric.get_name() + ' stdev'
    return name_line, name_list

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
