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
NUM_EPOCHS = 64
LEARNING_RATE = 1e-4

LOAD_MODEL = False
MODEL_PATH = Path("./trained_models/league_model")

INITIAL_MODEL_PATH = Path("./trained_models/base_model")  # Path to the initial EMModel checkpoint

## Dataset properties ##
TRAIN_DATASET_PATHS = ["data/nations_league_new.csv", "data/evaluation/wm18.csv", "data/evaluation/wm22.csv", "data/evaluation/em20.csv"]
VALIDATION_DATASET_PATH = ["data/evaluation/em24.csv"]
AVERAGE_PERFORMANCE_PATHS = [
    (2020, "data/4_2020"),
    (2024, "data/4_2024"),
    (2016, "data/4_2016"),
    (2022, "data/1_2022"),
    (2018, "data/1_2018"),
    (2018, "data/5_2018"),
    (2020, "data/5_2020"),
    (2022, "data/5_2022")
]
MAPPINGS_FILE_PATH_TRAIN = "data/mappings_without_names_train.json"
MAPPINGS_FILE_PATH_TEST = "data/mappings_without_names_test.json"
CATEGORICAL_COLUMNS = ["home/away", "player_name", "player_position", "league"]
DROP_COLUMNS = ["game_won", "rating", "team", "id"]

# Columns that are updated with the average performance data
COLUMNS_TO_UPDATE = ['minutes_played', 'attempted_shots', 'shots_on_goal', 'goals', 'assists',
                     'total_passes', 'key_passes', 'pass_completion', 'saves', 'tackles',
                     'blocks', 'interceptions', 'conceded_goals', 'total_duels', 'won_duels',
                     'attempted_dribbles', 'successful_dribbles', 'cards']

'''
Cross references the players in a given dataset with the avg_data dataset to find out which league they played in in a given season
This information is then appended to the given dataset
'''
def add_league_to_dataset(dataset, avg_data):
    missing_players = 0
    # iterate over the dataset and add the league to each player
    for index, row in dataset.iterrows():
        player_id = row['id']
        season = row['season']
        # if the player can be matched, add his league
        if player_id in avg_data[season]['player_id'].values:
            league = avg_data[season].loc[avg_data[season]['player_id'] == player_id, 'league'].values[0]
            dataset.loc[index, 'league'] = league
        elif row['player_name'] == 'puffer_player':
            # separate league for puffer players
            dataset.loc[index, 'league'] = 'puffer_league'
        else:
            # for missing players set league to unknown
            missing_players += 1
            dataset.loc[index, 'league'] = 'unknown'

    return dataset


'''
For all the matches in a give dataset replace the player stats with their average performance in the last year
(I merely moved this code snippet to a separate function so I could use it for multiple datasets, full credit for writing it to @Daniel)
'''
def replace_stats_with_avg(avg_data, data, columns_to_update):
    # iterate the avg_data
    for season, avg_data_season in avg_data.items():
        player_no_data = []

        # get the season data from the evaluation data (i.e. all the data from the given season)
        season_data = data.loc[data['season'] == season].copy()

        # create a nested tqdm subloop for the season data
        loop = tqdm(season_data.iterrows(), desc="Replacing player stats", total=season_data.shape[0], leave=False)
        loop.set_postfix(missing_players=0, season=season)

        # iterate over the season data from the dataset
        for index, row in loop:
            player_id = row['id']
            # replace the stats with the average performance if the player is in the avg_data
            if player_id in avg_data_season['player_id'].values:
                for column in columns_to_update:
                    season_data[column] = season_data[column].astype(float)
                    season_data.loc[index, column] = \
                    avg_data_season.loc[avg_data_season['player_id'] == player_id, column].values[0]
            # if the player is a puffer player, set the league to puffer_league
            elif row["player_name"] == "puffer_player":
                season_data.loc[index, 'league'] = "puffer_league"
            # if the player is not in the avg_data, add him to the player_no_data list
            else:
                if player_id not in player_no_data:
                    player_no_data.append(player_id)
                    loop.set_postfix(missing_players=len(player_no_data), season=season)

        # Replace the original data for the current season with the updated data
        for column in season_data.columns:
            data.loc[data['season'] == season, column] = season_data[column].astype(data[column].dtype)

    return data


def get_data_loader():
    # 1. Get the train data from the paths
    dataset_train = load_dataset(TRAIN_DATASET_PATHS)

    # Load the average performance data for all the relevant seasons
    avg_data = load_avg_performance_data(AVERAGE_PERFORMANCE_PATHS)
    # Add the league to all players in the train dataset
    dataset_train = add_league_to_dataset(dataset_train, avg_data)

    # Replace the training stats with the average performance TODO: this is experimental, depending on the training performance we might want to change this
    #dataset_train = replace_stats_with_avg(avg_data, dataset_train, COLUMNS_TO_UPDATE)

    # Preprocess the train data, i.e. map the categorical columns to integers, drop some columns, etc.
    dataset_train = preprocess_dataset(
        dataset_train,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TRAIN,
        DROP_COLUMNS,
        remove_player_names=True,
    )

    # create the Dataloader for the training data
    dataset_train = DatasetWithLeagues(dataset_train, normalize=True)
    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)


    # 2. Get the validation data from average performance and the given paths
    em_data = load_dataset(VALIDATION_DATASET_PATH)
    em_data = add_league_to_dataset(em_data, avg_data)

    # Replace the validation match stats with the average performance
    validation_data = replace_stats_with_avg(avg_data, em_data, COLUMNS_TO_UPDATE)

    # Preprocess the validation data
    validation_data = preprocess_dataset(
        validation_data,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TEST,
        DROP_COLUMNS + ["team"],
        remove_player_names=True
    )

    # Create a DataLoader for the validation data
    validation_data = DatasetWithLeagues(validation_data, normalize=True)
    data_loader_validation = DataLoader(validation_data, batch_size=32, shuffle=True)

    return data_loader_train, data_loader_validation, len(dataset_train.league_mapping)


def train(data_loader_train, data_loader_validation, num_leagues):
    # Initialize the model
    em_model = EMModelWithLeague(num_leagues=num_leagues).to(DEVICE)

    # Load the base model
    assert os.path.exists(
        INITIAL_MODEL_PATH), f"Initial model path {INITIAL_MODEL_PATH} does not exist. Please run training_step1.py first."  # fmt: skip

    # checking if the device is cpu, if so, we need to map the model to cpu
    if DEVICE == "cpu":
        initial_checkpoint = torch.load(INITIAL_MODEL_PATH, map_location=torch.device("cpu"))
    else:
        initial_checkpoint = torch.load(INITIAL_MODEL_PATH)

    em_model.load_state_dict(initial_checkpoint["state_dict"], strict=False)
    print("Loaded initial EMModel weights.")
    em_model.to(DEVICE)

    # do an evaluation of the model with the initial weights
    best_eval_loss, accu = em_model.eval_model(dataloader=data_loader_validation, device=DEVICE)
    print(f"Initial evaluation loss: {best_eval_loss}")
    print(f"Initial evaluation accuracy: {accu}")

    # Set model to train mode
    em_model.train()

    # Freeze the weights of the team classifier
    em_model.freeze_team_classifier()

    # Only optimize the leagueToScalar and game classifier parameters
    optimizer = torch.optim.Adam(
        list(em_model.leagueToScalar.parameters()) + list(em_model.gameClassifier.parameters()), lr=LEARNING_RATE
    )

    if LOAD_MODEL:
        assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist, but LOAD_MODEL is set to True."
        em_model.load_model(optimizer, LEARNING_RATE, MODEL_PATH)
        best_eval_loss, _ = em_model.eval_model(dataloader=data_loader_validation, device=DEVICE)

    best_eval_loss = None

    # TODO: pull this out of scope
    draw_threshold = 0.05

    # Save the accuracies and losses for plotting with their respective epochs
    save_accuracies_train = {}
    save_losses_train = {}

    save_accuracies_validation = {}
    save_losses_validation = {}
    for num_epoch in range(NUM_EPOCHS):
        curr_train_loss, curr_train_accuracy = em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            loss_fn=em_model.get_loss(),
            optimizer=optimizer,
            device=DEVICE,
            draw_threshold=draw_threshold
        )

        curr_eval_loss, curr_validation_accuracy = em_model.eval_model(dataloader=data_loader_validation, device=DEVICE, draw_threshold=draw_threshold)
        print(f"\nValidation Loss after epoch {num_epoch + 1}: {curr_eval_loss} and accuracy: {curr_validation_accuracy}\n")

        # save all the accuracies and losses for plotting
        save_accuracies_train[num_epoch] = curr_train_accuracy
        save_losses_train[num_epoch] = curr_train_loss

        save_accuracies_validation[num_epoch] = curr_validation_accuracy
        save_losses_validation[num_epoch] = curr_eval_loss

        if best_eval_loss is None or curr_eval_loss < best_eval_loss:
            em_model.save_model(optimizer, MODEL_PATH)
            best_eval_loss = curr_eval_loss


    # show the training and validation loss and accuracy
    plot_loss(save_losses_train, save_losses_validation)
    plot_accuracy(save_accuracies_train, save_accuracies_validation)


'''
Loads the average performance data from the given paths
Returns a dictionary with the year as key and a dataframe with the average stat data per player as value 
'''
def load_avg_performance_data(paths):
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
                        df = df.drop_duplicates(subset=['player_name'], keep='first')
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
