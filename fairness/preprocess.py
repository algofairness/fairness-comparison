import sys
import os
import pandas as pd
import fire
from fairness.data.objects.list import DATASETS, get_dataset_names

def prepare_data(dataset_names = get_dataset_names()):

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_names:
            continue
        print("--- Processing dataset: %s ---" % dataset.get_dataset_name())
        data_frame = dataset.load_raw_dataset()
        d = preprocess(dataset, data_frame)
        
        for k, v in d.items():
            write_to_file(dataset.get_filename(k), v)

def write_to_file(filename, dataframe):
    print("Writing data to: %s" % filename)
    dataframe.to_csv(filename, index = False)

def preprocess(dataset, data_frame):
    """
    The preprocess function takes a pandas data frame and returns two modified data frames:
    1) all the data as given with any features that should not be used for training or fairness
    analysis removed.
    2) only the numerical and ordered categorical data, sensitive attributes, and class attribute.
    Categorical attributes are one-hot encoded.
    3) the numerical data (#2) but with a binary (numerical) sensitive attribute
    """

    # Remove any columns not included in the list of features to keep.
    smaller_data = data_frame[dataset.get_features_to_keep()]

    # Handle missing data.
    missing_processed = dataset.handle_missing_data(smaller_data)

    # Remove any rows that have missing data.
    missing_data_removed = missing_processed.dropna()
    missing_data_count = missing_processed.shape[0] - missing_data_removed.shape[0]
    if missing_data_count > 0:
        print("Missing Data: " + str(missing_data_count) + " rows removed from dataset " +  \
              dataset.get_dataset_name())

    # Do any data specific processing.
    processed_data = dataset.data_specific_processing(missing_data_removed)

    print("\n-------------------")
    print("Balance statistics:")
    print("\nClass:")
    print(dataset.get_class_balance_statistics(processed_data))
    print("\nSensitive Attribute:")
    for r in dataset.get_sensitive_attribute_balance_statistics(processed_data):
        print(r)
        print("\n")
    print("\n")

    # Handle multiple sensitive attributes by creating a new attribute that's the joint distribution
    # of all of those attributes.  For example, if a dataset has both 'Race' and 'Gender', the
    # combined feature 'Race-Gender' is created that has attributes, e.g., 'White-Woman'.
    sensitive_attrs = dataset.get_sensitive_attributes()
    if len(sensitive_attrs) > 1:
        new_attr_name = '-'.join(sensitive_attrs)
        ## TODO: the below may fail for non-string attributes
        processed_data = processed_data.assign(temp_name =
                             processed_data[sensitive_attrs].apply('-'.join, axis=1))
        processed_data = processed_data.rename(columns = {'temp_name' : new_attr_name})
        # dataset.append_sensitive_attribute(new_attr_name)
        # privileged_joint_vals = '-'.join(dataset.get_privileged_class_names(""))
        # dataset.get_privileged_class_names("").append(privileged_joint_vals)

    # Create a one-hot encoding of the categorical variables.
    processed_numerical = pd.get_dummies(processed_data,
                                         columns = dataset.get_categorical_features())

    # Create a version of the numerical data for which the sensitive attribute is binary.
    sensitive_attrs = dataset.get_sensitive_attributes_with_joint()
    privileged_vals = dataset.get_privileged_class_names_with_joint("")
    processed_binsensitive = make_sensitive_attrs_binary(
        processed_numerical, sensitive_attrs, privileged_vals)

    # Create a version of the categorical data for which the sensitive attributes is binary.
    processed_categorical_binsensitive = make_sensitive_attrs_binary(
        processed_data, sensitive_attrs,
        dataset.get_privileged_class_names("")) ## FIXME
    # Make the class attribute numerical if it wasn't already (just for the bin_sensitive version).
    class_attr = dataset.get_class_attribute()
    pos_val = dataset.get_positive_class_val("") ## FIXME

    processed_binsensitive = make_class_attr_num(processed_binsensitive, class_attr, pos_val)

    return { "original": processed_data,
             "numerical": processed_numerical,
             "numerical-binsensitive": processed_binsensitive,
             "categorical-binsensitive": processed_categorical_binsensitive }

def make_sensitive_attrs_binary(dataframe, sensitive_attrs, privileged_vals):
    newframe = dataframe.copy()
    for attr, privileged in zip(sensitive_attrs, privileged_vals):
        # replace privileged vals with 1
        newframe[attr] = newframe[attr].replace({ privileged : 1 })
        # replace all other vals with 0
        newframe[attr] = newframe[attr].replace("[^1]", 0, regex = True)
    return newframe

def make_class_attr_num(dataframe, class_attr, positive_val):
    # don't change the class attribute unless its a string (pandas type: object)
    if (dataframe[class_attr].dtypes == 'object'):
        dataframe[class_attr] = dataframe[class_attr].replace({ positive_val : 1 })
        dataframe[class_attr] = dataframe[class_attr].replace("[^1]", 0, regex = True)
    return dataframe

def main():
    fire.Fire(prepare_data)

if __name__ == '__main__':
    main()
