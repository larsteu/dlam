import os.path

import pandas as pd
import json

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def special_transforms(df):
    df["pass_completion"] = df["pass_completion"].apply(lambda x: int(x.replace("%", "")) if type(x) is str else x)
    return df


def load_dataset(dataset_path_list):
    dataset = special_transforms(pd.read_csv(dataset_path_list[0]))
    for i in range(1, len(dataset_path_list)):
        df = pd.read_csv(dataset_path_list[i])
        df = special_transforms(df)
        dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset


def preprocess_dataset(
    dataset: pd.DataFrame,
    categorical_columns,
    mappings_file_path,
    columns_to_drop,
    remove_player_names=False,
):
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


def normalize_dataset(dataset, normalization_info_file_path, use_existing_normalisation=False):
    normalization_info = {}
    loop = tqdm(dataset.columns)
    loop.set_description(f"Normalizing data")

    if use_existing_normalisation:
        for i, col in enumerate(loop):
            with open(normalization_info_file_path) as json_file:
                normalization_info = json.load(json_file)
            max_val = normalization_info[col]["max"]
            min_val = normalization_info[col]["min"]

            if max_val - min_val == 0:
                dataset[col] = 0
            else:
                dataset[col] = (dataset[col] - min_val) / (max_val - min_val)
        return dataset

    for i, col in enumerate(loop):
        max_val = int(dataset[col].max())
        min_val = int(dataset[col].min())

        values = {"max": max_val, "min": min_val}
        normalization_info[col] = values

        if max_val - min_val == 0:
            dataset[col] = 0
        else:
            dataset[col] = (dataset[col] - min_val) / (max_val - min_val)

    with open(normalization_info_file_path, "w") as outfile:
        json.dump(normalization_info, outfile)

    return dataset


def plot_loss(train_loss, val_loss, title="Loss"):
    if train_loss is None:
        train_loss = {epoch: 0 for epoch in val_loss.keys()}
    epochs = list(train_loss.keys())

    if train_loss is not None:
        plt.plot(epochs, list(train_loss.values()), label="Train loss")
    plt.plot(epochs, list(val_loss.values()), label="Validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(train_accuracy, val_accuracy, title="Accuracy"):
    if train_accuracy is None:
        train_accuracy = {epoch: 0 for epoch in val_accuracy.keys()}
    epochs = list(train_accuracy.keys())

    if train_accuracy is not None:
        plt.plot(epochs, list(train_accuracy.values()), label="Train accuracy")
    plt.plot(epochs, list(val_accuracy.values()), label="Validation accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def calculate_correct_predictions(outputs, target, return_tensor=False):
    """
    The calculate_correct_predictions method calculates the number of correct predictions made by the model.
    The method takes the model outputs, the target values, and an optional draw_threshold parameter.
    The draw_threshold parameter is used to determine when the model predicts a draw.
    If the absolute difference between the predicted goals scored by the two teams is less than the draw_threshold, the model predicts a draw.
    The method returns the number of correct predictions made by the model.
    """
    correct_predictions = 0

    for i in range(len(outputs)):
        # check which output has the highest value
        if outputs[i][0] > outputs[i][1] and outputs[i][0] > outputs[i][2]:
            if target[i][0] == 1:
                correct_predictions += 1
        elif outputs[i][1] > outputs[i][0] and outputs[i][1] > outputs[i][2]:
            if target[i][1] == 1:
                correct_predictions += 1
        elif outputs[i][2] > outputs[i][0] and outputs[i][2] > outputs[i][1]:
            if target[i][2] == 1:
                correct_predictions += 1

    total_predictions = len(outputs)
    return correct_predictions, total_predictions
