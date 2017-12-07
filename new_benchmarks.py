
import fire
from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS

NUM_TRIALS_DEFAULT = 10

def run(num_trials = NUM_TRIALS_DEFAULT, dataset_names = get_dataset_names()):
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_names:
            continue
        processed_dataset = ProcessedData(dataset)
        processed_splits, numerical_splits = processed_dataset.create_train_test_splits(num_trials)
        for algorithm in ALGORITHMS:
            for i in range(0, num_trials):
                train, test = processed_splits[i]
                algorithm.run(train, test, dataset.get_sensitive_attrs, params)

                for metric in METRICS:
                    # write metric, algorithm to dataset file
                    pass

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()
