import os.path

import pandas as pd
import json
from tqdm import tqdm

DATASET_PATH = ['./data/bundesliga_matches.csv',
                './data/la_liga_matches.csv',
                './data/ligue_1_matches.csv',
                './data/premier_league_matches.csv',
                './data/serie_a_matches.csv',
                './data/top5_2022-23.csv',
                './data/21-22/la_liga_matches_2021-22.csv',
                './data/21-22/premier_league_matches_2021-22.csv']
MAPPINGS_FILE_PATH = './data/mappings.json'
CATEGORICAL_COLUMNS = ['home/away',
                       'player_name',
                       'player_position',
                       'game_won']


def load_dataset():
    dataset = pd.read_csv(DATASET_PATH[0])
    for i in range(1, len(DATASET_PATH)):
        df = pd.read_csv(DATASET_PATH[i])
        dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset


def preprocess_dataset(dataset: pd.DataFrame):
    dataset.drop(columns=['Unnamed: 0', 'game_result'], inplace=True)
    processed_dataset = categories_to_numerical(dataset, CATEGORICAL_COLUMNS)
    return processed_dataset


def categories_to_numerical(dataset: pd.DataFrame, cat_cols):
    if os.path.exists(MAPPINGS_FILE_PATH):
        with open(MAPPINGS_FILE_PATH) as json_file:
            mappings_file = json.load(json_file)

        loop = tqdm(range(1))
        loop.set_description(f"Converting categorical values to numerical (using existing mapping file)")

        for i, _ in enumerate(loop):
            dataset = dataset.replace(mappings_file)
            dataset = dataset.fillna(mappings_file["player_position"]["NaN"])

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

    with open(MAPPINGS_FILE_PATH, "w") as outfile:
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

        dataset[col] = (dataset[col] - min_val) / (max_val - min_val)

    with open(normalization_info_file_path, "w") as outfile:
        json.dump(normalization_info, outfile)

    return dataset
