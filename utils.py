import os.path

import pandas as pd
import json
from tqdm import tqdm


def special_transforms(df):
    df['pass_completion'] = df['pass_completion'].apply(lambda x: int(x.replace('%', '')) if type(x) is str else x)
    return df


def load_dataset(dataset_path_list):
    dataset = pd.read_csv(dataset_path_list[0])
    for i in range(1, len(dataset_path_list)):
        df = pd.read_csv(dataset_path_list[i])
        df = special_transforms(df)
        dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset


def preprocess_dataset(dataset: pd.DataFrame, categorical_columns, mappings_file_path, columns_to_drop,
                       remove_player_names=False):
    dataset.drop(columns=columns_to_drop, inplace=True)
    if remove_player_names:
        dataset["player_name"] = "Player"

    processed_dataset = categories_to_numerical(dataset, categorical_columns, mappings_file_path)
    return processed_dataset


def categories_to_numerical(dataset: pd.DataFrame, cat_cols, mappings_file_path):
    if os.path.exists(mappings_file_path):
        with open(mappings_file_path) as json_file:
            mappings_file = json.load(json_file)

        loop = tqdm(range(1))
        loop.set_description(f"Converting categorical values to numerical (using existing mapping file)")

        for i, _ in enumerate(loop):
            dataset = dataset.replace(mappings_file)

        return dataset

    mappings = {}

    loop = tqdm(cat_cols)
    loop.set_description(f"Converting categorical values to numerical")

    for i, col in enumerate(loop):
        vals = dataset[col].unique()
        counter = 0
        mapping = {}

        for val in vals:
            mapping[val] = counter
            dataset = dataset.replace(val, counter)
            counter = counter + 1

        mappings[col] = mapping

    with open(mappings_file_path, "w") as outfile:
        json.dump(mappings, outfile)

    return dataset


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
    loop = tqdm(dataset.columns)
    loop.set_description(f"Normalizing data")
    for i, col in enumerate(loop):
        max_val = int(dataset[col].max())
        min_val = int(dataset[col].min())

        values = {"max": max_val, "min": min_val}
        normalization_info[col] = values

        if max_val-min_val == 0:
            dataset[col] = 0
        else:
            dataset[col] = (dataset[col] - min_val) / (max_val - min_val)

    with open(normalization_info_file_path, "w") as outfile:
        json.dump(normalization_info, outfile)

    return dataset
