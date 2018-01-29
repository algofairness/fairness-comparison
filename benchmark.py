import fire
import os
import statistics

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS
from metrics.list import METRICS

NUM_TRIALS_DEFAULT = 10

def get_algorithm_names():
    return [algorithm.get_name() for algorithm in ALGORITHMS]

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names()):
    algorithms_to_run = algorithm
    metrics_list = get_metrics_list()

    print("Datasets: '%s'" % dataset)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = ProcessedData(dataset_obj)
        processed_splits, numerical_splits, binsensitive_splits =  \
            processed_dataset.create_train_test_splits(num_trials)

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        print(all_sensitive_attributes)
        for sensitive in all_sensitive_attributes:

            print("Sensitive attribute:" + sensitive)

            f_bin = create_detailed_file(
                        dataset_obj.get_results_numerical_binsensitive_filename(sensitive))
            f_num = create_detailed_file(
                        dataset_obj.get_results_numerical_filename(sensitive))
            f_all = create_detailed_file(
                        dataset_obj.get_results_filename(sensitive))

            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm:" + algorithm.get_name())

                for i in range(0, num_trials):

                    # Run on data that is all numerical, binary sensitive attributes, numerical
                    # class data.  We assume all algorithms can handle this type of data.
                    train, test = binsensitive_splits[i]
                    params, results = run_eval_alg(algorithm, train, test, dataset_obj,
                                                   all_sensitive_attributes, sensitive, True)
                    write_alg_results(f_bin, algorithm.get_name(), params, results)

                    if not algorithm.binary_sensitive_attrs_only():
                        # Run on data that is all numerical except for the sensitive attributes
                        # and class attribute.
                        train, test = numerical_splits[i]
                        params, results = run_eval_alg(algorithm, train, test, dataset_obj,
                                                       all_sensitive_attributes, sensitive, False)
                        write_alg_results(f_num, algorithm.get_name(), params, results)

                        if not algorithm.numerical_data_only():
                            # Run on data that may be numerical or categorical.
                            train, test = processed_splits[i]
                            params, results = run_eval_alg(algorithm, train, test, dataset_obj,
                                                           all_sensitive_attributes, sensitive,
                                                           False)
                            write_alg_results(f_all, algorithm.get_name(), params, results)

        print("Results written to:")
        print("    " + dataset_obj.get_results_filename(sensitive))
        print("    " + dataset_obj.get_results_numerical_filename(sensitive))
        print("    " + dataset_obj.get_results_numerical_binsensitive_filename(sensitive))

        f_bin.close()
        f_num.close()
        f_all.close()

def write_alg_results(file_handle, alg_name, params, results_list):
    line = alg_name + ','
    line += str(params) + ','  ## TODO: do something more parseable with multiple params here
    line += ','.join(str(x) for x in results_list) + '\n'
    file_handle.write(line)
    # Make sure the file is written to disk line-by-line:
    file_handle.flush()
    os.fsync(file_handle.fileno())

def run_eval_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
                 bin_sensitive):
    """
    Runs the algorithm and gets the resulting metric evaluations.
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint()
    positive_val = dataset.get_positive_class_val()
    if bin_sensitive:
        # in this case, the real data has been overwritten with 0/1
        privileged_vals = [1 for x in all_sensitive_attributes]
        positive_val = 1

    actual, predicted, sensitive, params =  \
        run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
                privileged_vals, positive_val)

    one_run_results = []
    for metric in METRICS:
        result = metric.calc(actual, predicted, sensitive, privileged_vals, positive_val)
        one_run_results.append(result)

    return params, one_run_results

def run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
            privileged_vals, positive_val):
    class_attr = dataset.get_class_attribute()
    params = algorithm.get_default_params()

    # get the actual classifications and sensitive attributes
    actual = test[class_attr]
    sensitive = test[single_sensitive].values.tolist()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.
    predictions = algorithm.run(train, test, class_attr, positive_val, all_sensitive_attributes,
                                single_sensitive, privileged_vals, params)

    return actual, predictions, sensitive, params

def get_metrics_list():
    return [metric.get_name() for metric in METRICS]

def get_detailed_metrics_header():
    return ','.join(['algorithm', 'params'] + get_metrics_list())

def create_detailed_file(filename):
    f = open(filename, 'w')
    f.write(get_detailed_metrics_header() + '\n')
    return f

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
