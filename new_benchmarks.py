import fire
from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS
from metrics.list import METRICS

NUM_TRIALS_DEFAULT = 10
RESULT_DIR = "results/"

def run(num_trials = NUM_TRIALS_DEFAULT, dataset_names = get_dataset_names()):
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_names:
            continue
        print("\nEvaluating dataset:" + dataset.get_dataset_name())

        processed_dataset = ProcessedData(dataset)
        processed_splits, numerical_splits = processed_dataset.create_train_test_splits(num_trials)

        for algorithm in ALGORITHMS:
            for i in range(0, num_trials):
                if not algorithm.numerical_data_only():
                    train, test = processed_splits[i]
                    actual, predicted, sensitive = run_alg(algorithm, train, test, dataset)
                train, test = numerical_splits[i]
                actual_num, predicted_num, sensitive_num = run_alg(algorithm, train, test, dataset)

                for Metric in METRICS:
                    m = Metric(actual_num, predicted_num)
                    result = m.calc() 
                    print(m.get_metric_name() + ':' + str(result))
                    ## TODO: save metric so that avg and stddev can be calculated
            ## TODO: calculate avg and stddev per metric
            ## TODO: write avg and stddev per metric to file

def run_alg(algorithm, train, test, dataset):
    class_attr = dataset.get_class_attribute()
    sensitive_attrs = dataset.get_sensitive_attributes()
    params = {}
    return algorithm.run(train, test, class_attr, sensitive_attrs, params)

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
