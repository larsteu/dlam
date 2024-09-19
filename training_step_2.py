import json
import os.path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.model_with_leagues import EMModelWithLeague
from dataset import DatasetWithLeagues
from utils import load_dataset, preprocess_dataset, plot_loss, plot_accuracy
from pathlib import Path
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Model settings ##
NUM_EPOCHS = 1024
LEARNING_RATE = 1e-4

LOAD_MODEL = False
MODEL_PATH = Path("./trained_models/league_model")

INITIAL_MODEL_PATH = Path("./trained_models/base_model")  # Path to the initial EMModel checkpoint

# New constants for saved data
PREPROCESSED_TRAIN_DATA_PATH = "data/preprocessed_train_data.csv"
PREPROCESSED_VALIDATION_DATA_PATH = "data/preprocessed_validation_data.csv"

## Dataset properties ##
TRAIN_DATASET_PATHS = [
    "data/nations_league_new.csv",
    "data/evaluation/em20.csv",
    "data/evaluation/wm18.csv",
    "data/evaluation/wm22.csv",
]
VALIDATION_DATASET_PATH = [
    "data/evaluation/em24.csv",
]
AVERAGE_PERFORMANCE_PATHS = [
    (2020, "data/4_2020"),
    (2024, "data/4_2024"),
    (2016, "data/4_2016"),
    (2022, "data/1_2022"),
    (2018, "data/1_2018"),
    (2018, "data/5_2018"),
    (2020, "data/5_2020"),
    (2022, "data/5_2022"),
]
MAPPINGS_FILE_PATH_TRAIN = "data/mappings_without_names_train.json"
MAPPINGS_FILE_PATH_TEST = "data/mappings_without_names_test.json"
CATEGORICAL_COLUMNS = ["home/away", "player_name", "player_position"]
DROP_COLUMNS = ["game_won", "rating", "team", "id", "season"]

# Columns that are updated with the average performance data
COLUMNS_TO_UPDATE = [
    "minutes_played",
    "attempted_shots",
    "shots_on_goal",
    "goals",
    "assists",
    "total_passes",
    "key_passes",
    "pass_completion",
    "saves",
    "tackles",
    "blocks",
    "interceptions",
    "conceded_goals",
    "total_duels",
    "won_duels",
    "attempted_dribbles",
    "successful_dribbles",
    "cards",
]

"""
Cross references the players in a given dataset with the avg_data dataset to find out which league they played in in a given season
This information is then appended to the given dataset
"""
def add_league_to_dataset(dataset, avg_data):
    missing_players = 0
    # iterate over the dataset and add the league to each player
    for index, row in dataset.iterrows():
        player_id = row["id"]
        season = row["season"]
        # if the player can be matched, add his league
        if player_id in avg_data[season]["player_id"].values:
            league = avg_data[season].loc[avg_data[season]["player_id"] == player_id, "league"].values[0]
            dataset.loc[index, "league"] = league
        elif row["player_name"] == "puffer_player":
            # separate league for puffer players
            dataset.loc[index, "league"] = "puffer_league"
        else:
            # for missing players set league to unknown
            missing_players += 1
            dataset.loc[index, "league"] = "unknown"

    return dataset


def replace_stats_with_avg(avg_data, data, columns_to_update):
    """
    For all the matches in a give dataset replace the player stats with their average performance in the last year
    (I merely moved this code snippet to a separate function so I could use it for multiple datasets, full credit for writing it to @Daniel)
    """
    # iterate the avg_data
    for season, avg_data_season in avg_data.items():
        player_no_data = []

        # get the season data from the evaluation data (i.e. all the data from the given season)
        season_data = data.loc[data["season"] == season].copy()

        # create a nested tqdm subloop for the season data
        loop = tqdm(season_data.iterrows(), desc="Replacing player stats", total=season_data.shape[0], leave=False)
        loop.set_postfix(missing_players=0, season=season)

        # iterate over the season data from the dataset
        for index, row in loop:
            player_id = row["id"]
            # replace the stats with the average performance if the player is in the avg_data
            if player_id in avg_data_season["player_id"].values:
                for column in columns_to_update:
                    season_data[column] = season_data[column].astype(float)
                    season_data.loc[index, column] = avg_data_season.loc[
                        avg_data_season["player_id"] == player_id, column
                    ].values[0]
            # if the player is a puffer player, set the league to puffer_league
            elif row["player_name"] == "puffer_player":
                season_data.loc[index, "league"] = "puffer_league"
            # if the player is not in the avg_data, add him to the player_no_data list
            else:
                if player_id not in player_no_data:
                    player_no_data.append(player_id)
                    loop.set_postfix(missing_players=len(player_no_data), season=season)

        # Replace the original data for the current season with the updated data
        for column in season_data.columns:
            data.loc[data["season"] == season, column] = season_data[column].astype(data[column].dtype)

    return data


def save_preprocessed_data(data, file_path):
    data.to_csv(file_path, index=False)
    print(f"Saved preprocessed data to {file_path}")


def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)


def preprocess_or_load_data(dataset_paths, avg_data, columns_to_update, output_path):
    if os.path.exists(output_path):
        print(f"Loading preprocessed data from {output_path}")
        return load_preprocessed_data(output_path)

    print(f"Preprocessing data for {output_path}")
    dataset = load_dataset(dataset_paths)
    dataset = add_league_to_dataset(dataset, avg_data)
    dataset = replace_stats_with_avg(avg_data, dataset, columns_to_update)
    return dataset



def get_leagues_mapping(data):
    leagues = data["league"]

    # process them by removing all dots, commas and spaces and make them lowercase
    leagues = [league.replace(".", "").replace(",", "").replace(" ", "").lower() for league in leagues]

    unique_leagues = set(leagues)

    # iterate over the unique leagues and create a mapping
    league_mapping = {}
    for idx, league in enumerate(unique_leagues):
        league_mapping[league] = leagues.count(league)

    # sort the mapping by the number of occurences
    league_mapping = dict(sorted(league_mapping.items(), key=lambda item: item[1], reverse=True))

    return league_mapping

def get_data_loader():
    # Load the average performance data for all the relevant seasons
    avg_data = load_avg_performance_data(AVERAGE_PERFORMANCE_PATHS)

    # Preprocess and save/load training data
    dataset_train = preprocess_or_load_data(TRAIN_DATASET_PATHS, avg_data, COLUMNS_TO_UPDATE,
                                             PREPROCESSED_TRAIN_DATA_PATH)

    # Preprocess and save/load validation data
    validation_data = preprocess_or_load_data(VALIDATION_DATASET_PATH, avg_data, COLUMNS_TO_UPDATE,
                                               PREPROCESSED_VALIDATION_DATA_PATH)

    num_mapped_leagues = 35 # when you change this, you have to re-run the preprocessing
    if not (os.path.exists(PREPROCESSED_TRAIN_DATA_PATH) and os.path.exists(PREPROCESSED_VALIDATION_DATA_PATH)):
        # get the combined league mapping
        league_mapping = get_leagues_mapping(dataset_train)
        league_mapping.update(get_leagues_mapping(validation_data))

        # sort the league mapping by the number of occurences
        league_mapping = dict(sorted(league_mapping.items(), key=lambda item: item[1], reverse=True))

        # go over the league mapping and assign an index to the top N leagues (rest will be assigned to "other")
        for idx, league in enumerate(list(league_mapping.keys())):
            if idx < num_mapped_leagues:
                league_mapping[league] = idx
            else:
                league_mapping[league] = num_mapped_leagues

        # replace the league names with the assigned index (the league has to be pre_processed first, i.e. removing dots, commas and spaces and making it lowercase)
        dataset_train["league"] = dataset_train["league"].apply(lambda x: league_mapping[x.replace(".", "").replace(",", "").replace(" ", "").lower()])
        validation_data["league"] = validation_data["league"].apply(lambda x: league_mapping[x.replace(".", "").replace(",", "").replace(" ", "").lower()])

        # save the data with the updated league mapping, unless it already exists
        save_preprocessed_data(dataset_train, PREPROCESSED_TRAIN_DATA_PATH)
        save_preprocessed_data(validation_data, PREPROCESSED_VALIDATION_DATA_PATH)

        # save the league mapping to a file
        with open("data/league_mapping.json", "w") as f:
            json.dump(league_mapping, f)

    # Preprocess the train data
    dataset_train = preprocess_dataset(
        dataset_train,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TRAIN,
        DROP_COLUMNS,
        remove_player_names=True,
    )

    # Create the Dataloader for the training data
    dataset_train = DatasetWithLeagues(dataset_train, normalize=True)
    data_loader_train = DataLoader(dataset_train, batch_size=12, shuffle=True)

    # Preprocess the validation data
    validation_data = preprocess_dataset(
        validation_data,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TEST,
        DROP_COLUMNS + ["team"],
        remove_player_names=True,
    )

    # Create a DataLoader for the validation data
    validation_data = DatasetWithLeagues(validation_data, normalize=True)
    data_loader_validation = DataLoader(validation_data, batch_size=12, shuffle=False)

    return data_loader_train, data_loader_validation, num_mapped_leagues


def train(data_loader_train, data_loader_validation, num_leagues):
    num_leagues += 1  # Add one for the "other" league
    # Initialize the model
    em_model = EMModelWithLeague(num_leagues=num_leagues).to(DEVICE)

    # Load the base model
    assert os.path.exists(
        INITIAL_MODEL_PATH), f"Initial model path {INITIAL_MODEL_PATH} does not exist. Please run training_step1.py first."

    # Load the initial model weights
    initial_checkpoint = torch.load(INITIAL_MODEL_PATH, map_location=DEVICE)
    em_model.load_state_dict(initial_checkpoint["state_dict"], strict=False)
    print("Loaded initial EMModel weights.")

    # Set model to train mode
    em_model.train()

    # Freeze the weights of the team classifier
    em_model.freeze_base_model()

    # Only optimize the leagueToScalar and game classifier parameters
    optimizer = torch.optim.Adam(
        list(em_model.league_embedding.parameters()) + list(em_model.gameClassifier.parameters()),
        lr=LEARNING_RATE
    )

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        em_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded pre-trained model from {MODEL_PATH}")

    # Training loop
    save_accuracies_train, save_losses_train = {}, {}
    save_accuracies_validation, save_losses_validation = {}, {}

    best_eval_loss = None
    best_eval_accuracy = None

    for num_epoch in range(NUM_EPOCHS):
        em_model.train()
        # Train
        curr_train_loss, curr_train_accuracy = em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            loss_fn=em_model.get_loss(),
            optimizer=optimizer,
            device=DEVICE,
        )

        em_model.eval()
        # Evaluate
        curr_eval_loss, curr_validation_accuracy = em_model.eval_model(
            dataloader=data_loader_validation, device=DEVICE,
        )
        print(f"Epoch {num_epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {curr_train_loss:.4f}, Train Accuracy: {curr_train_accuracy:.4f}")
        print(f"Val Loss: {curr_eval_loss:.4f}, Val Accuracy: {curr_validation_accuracy:.4f}")

        # Save metrics
        save_accuracies_train[num_epoch] = curr_train_accuracy
        save_losses_train[num_epoch] = curr_train_loss
        save_accuracies_validation[num_epoch] = curr_validation_accuracy
        save_losses_validation[num_epoch] = curr_eval_loss

        # Save best model
        if best_eval_loss is None or curr_eval_loss < best_eval_loss:
            best_eval_loss = curr_eval_loss
            torch.save({
                'epoch': num_epoch,
                'model_state_dict': em_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_eval_loss,
            }, MODEL_PATH)
            print(f"Saved new best model with validation loss: {best_eval_loss:.4f}")

        # Save best accuracy model
        if best_eval_accuracy is None or curr_validation_accuracy >= best_eval_accuracy:
            best_eval_accuracy = curr_validation_accuracy
            torch.save({
                'epoch': num_epoch,
                'model_state_dict': em_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_eval_loss,
            }, Path(MODEL_PATH.__str__() + '_accuracy'))
            print(f"Saved new best model with validation accuracy: {best_eval_accuracy:.4f}")

    # Plot training results
    plot_loss(save_losses_train, save_losses_validation)
    plot_accuracy(save_accuracies_train, save_accuracies_validation)

    # Final evaluation
    em_model.eval()
    final_loss, final_accuracy = em_model.eval_model(dataloader=data_loader_validation, device=DEVICE)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

    # best model
    print(f"Best model - Accuracy: {best_eval_accuracy:.4f}")

def load_avg_performance_data(paths):
    """
    Loads the average performance data from the given paths
    Returns a dictionary with the year as key and a dataframe with the average stat data per player as value
    """
    avg_data = {}
    for year, path in paths:
        dataframes = []
        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)
                # Check if the file is not empty
                if os.path.getsize(file_path) > 0:
                    try:
                        df = pd.read_csv(file_path)
                        # Remove duplicates in file
                        df = df.drop_duplicates(subset=["player_name"], keep="first")
                        # Only append non-empty dataframes
                        if df.shape[0] > 0:
                            dataframes.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"Skipping empty file: {file}")

        # Always append year data
        if year in avg_data:
            avg_data[year] = pd.concat([avg_data[year], pd.concat(dataframes)])
        else:
            avg_data[year] = pd.concat(dataframes)

    return avg_data


if __name__ == "__main__":
    data_loader_train, data_loader_test, num_leagues = get_data_loader()
    train(data_loader_train, data_loader_test, num_leagues)
