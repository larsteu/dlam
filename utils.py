import pandas as pd
import json

DATASET_PATH = ['./data/bundesliga_matches.csv',
                './data/la_liga_matches.csv',
                './data/ligue_1_matches.csv',
                './data/premier_league_matches.csv',
                './data/serie_a_matches.csv']
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
