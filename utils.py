import pandas as pd
import json

DATASET_PATH = ''  # TODO
MAPPINGS_FILE_PATH = ''  # TODO
CATEGORICAL_COLUMNS = []  # TODO
TARGET_COLUMN = []  # TODO
COLUMN_MAP = {}  # TODO
COLUMNS_WITHOUT_TARGET = []  # TODO


def load_dataset():
    return pd.read_csv(DATASET_PATH, quotechar='"', quoting=1)


def preprocess_dataset(dataset):
    processed_dataset, mappings = categories_to_numerical(dataset, CATEGORICAL_COLUMNS)
    with open(MAPPINGS_FILE_PATH, "w") as outfile:
        json.dump(mappings, outfile)

    return processed_dataset


def categories_to_numerical(dataset, cat_cols):
    mappings = {}
    for col in cat_cols:
        vals = dataset[col].unique()
        counter = 0
        mapping = {}
        for val in vals:
            mapping[val] = counter
            dataset = dataset.replace(val, counter)
            counter = counter + 1
        mappings[col] = mapping

    return dataset, mappings


def numerical_to_categories(dataset, mapping_file):
    with open(mapping_file) as json_file:
        mappings = json.load(json_file)
    for column_name in mappings:
        for categorical_value in mappings[column_name]:
            dataset[column_name] = dataset[column_name].replace(
                mappings[column_name][categorical_value], categorical_value
            )

    return dataset


def denormalize_dataset(dataset, normalization_info_file):
    with open(normalization_info_file) as json_file:
        normalization_info = json.load(json_file)

    for column_name in normalization_info:
        max_val = normalization_info[column_name]["max"]
        min_val = normalization_info[column_name]["min"]

        dataset[column_name] = dataset[column_name] * (max_val - min_val) + min_val
    return dataset


def normalize_dataset(dataset, normalization_info_file_path):
    normalization_info = {}
    for col in dataset.columns:
        max_val = int(dataset[col].max())
        min_val = int(dataset[col].min())

        values = {"max": max_val, "min": min_val}
        normalization_info[col] = values

        dataset[col] = (dataset[col] - min_val) / (max_val - min_val)

    with open(normalization_info_file_path, "w") as outfile:
        json.dump(normalization_info, outfile)

    return dataset
