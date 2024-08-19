import os.path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.model_with_leagues import EMModelWithLeague
from dataset import DatasetWithLeagues
from utils import load_dataset, preprocess_dataset
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Model settings ##
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

LOAD_MODEL = False
MODEL_PATH = Path("./trained_models/league_model")

INITIAL_MODEL_PATH = Path("./trained_models/base_model")  # Path to the initial EMModel checkpoint

## Dataset properties ##
TRAIN_DATASET_PATH = ["data/evaluation/nations_league_with_league.csv"]
VALIDATION_DATASET_PATH = ["data/evaluation/em20.csv"]
AVERAGE_PERFORMANCE_2020_DATA_PATH = ["data/4_2020"]
AVERAGE_PERFORMANCE_2024_DATA_PATH = ["data/4_2024"]
MAPPINGS_FILE_PATH_TRAIN = "data/mappings_without_names_train.json"
MAPPINGS_FILE_PATH_TEST = "data/mappings_without_names_test.json"
CATEGORICAL_COLUMNS = ["home/away", "player_name", "player_position", "league"]
DROP_COLUMNS = ["game_won", "rating"]


def get_data_loader():
    # 1. Train data
    dataset_train = load_dataset(TRAIN_DATASET_PATH)
    dataset_train = preprocess_dataset(
        dataset_train,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TRAIN,
        DROP_COLUMNS,
        remove_player_names=True,
    )
    dataset_train = DatasetWithLeagues(dataset_train, normalize=True)
    data_loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

    # 2. Validation data from average performance
    avg_data = load_avg_performance_data(AVERAGE_PERFORMANCE_2020_DATA_PATH)
    em_data = load_dataset(VALIDATION_DATASET_PATH)

    columns_to_update = ['minutes_played', 'attempted_shots', 'shots_on_goal', 'goals', 'assists',
                         'total_passes', 'key_passes', 'pass_completion', 'saves', 'tackles',
                         'blocks', 'interceptions', 'conceded_goals', 'total_duels', 'won_duels',
                         'attempted_dribbles', 'successful_dribbles', 'cards']
    player_no_data = []

    # Ensure 'league' column exists in em_data
    if 'league' not in em_data.columns:
        em_data['league'] = 'unknown'
   
    for season, avg_data_season in avg_data.items():
        print("Aktuelle Saison: ", season)
        # get the season data from the evaluation data (i.e. all the data from the given season)
        season_data = em_data.loc[em_data['season'] == season].copy()

        # Ensure 'league' column exists in season_data
        if 'league' not in season_data.columns:
            season_data['league'] = 'unknown'

        for index, row in season_data.iterrows():
            player_id = row['id']

            if player_id in avg_data_season['player_id'].values:
                for column in columns_to_update:
                    season_data[column] = season_data[column].astype(float)
                    season_data.loc[index, column] = avg_data_season.loc[avg_data_season['player_id'] == player_id, column].values[0]

                # Update 'league' column
                season_data.loc[index, 'league'] = avg_data_season.loc[avg_data_season['player_id'] == player_id, 'league'].values[0]

            elif row["player_name"] == "puffer_player":
                season_data.loc[index, 'league'] = "puffer_league"
            else:
                if player_id not in player_no_data:
                    print(f"Player {row["player_name"]} not found in average performance data.")
                    player_no_data.append(player_id)

        print(f"Players not found in season {season}: {len(player_no_data)}")

        # Replace the original data for the current season with the updated data
        for column in season_data.columns:
            em_data.loc[em_data['season'] == season, column] = season_data[column].astype(em_data[column].dtype)


    validation_data = em_data

    # Preprocess the validation data
    validation_data = preprocess_dataset(
        validation_data,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TEST,
        DROP_COLUMNS+["team"],
        remove_player_names=True
    )

    # Create a DataLoader for the validation data
    validation_data = DatasetWithLeagues(validation_data, normalize=True)
    data_loader_validation = DataLoader(validation_data, batch_size=64, shuffle=True)

    # 3. Number of leagues & return
    num_leagues = len(dataset_train.league_mapping)

    return data_loader_train, data_loader_validation, num_leagues


def train(data_loader_train, data_loader_validation, num_leagues):
    em_model = EMModelWithLeague(num_leagues=num_leagues).to(DEVICE)

    # Load the base model
    assert os.path.exists(INITIAL_MODEL_PATH), f"Initial model path {INITIAL_MODEL_PATH} does not exist. Please run training_step1.py first."  # fmt: skip

    # checking if the device is cpu, if so, we need to map the model to cpu
    if DEVICE == "cpu":
        initial_checkpoint = torch.load(INITIAL_MODEL_PATH, map_location=torch.device("cpu"))
    else:
        initial_checkpoint = torch.load(INITIAL_MODEL_PATH)

    em_model.load_state_dict(initial_checkpoint["state_dict"], strict=False)
    print("Loaded initial EMModel weights.")

    # Freeze the weights of the team classifier
    em_model.freeze_team_classifier()

    # Only optimize the league embedding and game classifier parameters
    optimizer = torch.optim.Adam(
        list(em_model.leagueEmbedding.parameters()) + list(em_model.gameClassifier.parameters()), lr=LEARNING_RATE
    )

    if LOAD_MODEL:
        assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist, but LOAD_MODEL is set to True."
        em_model.load_model(optimizer, LEARNING_RATE, MODEL_PATH)
        best_eval_loss = em_model.eval_model(dataloader=data_loader_validation, device=DEVICE)

    best_eval_loss = None

    for num_epoch in range(NUM_EPOCHS):
        em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            loss_fn=em_model.get_loss(),
            optimizer=optimizer,
            device=DEVICE,
        )

        curr_eval_loss = em_model.eval_model(dataloader=data_loader_validation, device=DEVICE)
        print(f"Validation Loss after epoch {num_epoch + 1}: {curr_eval_loss}")

        if best_eval_loss is None or curr_eval_loss < best_eval_loss:
            em_model.save_model(optimizer, MODEL_PATH)
            best_eval_loss = curr_eval_loss


def load_avg_performance_data(paths):
    avg_data = {}
    for path in paths:
        year = path.split('_')[-1]
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

        avg_data[int(year)] = pd.concat(dataframes)
    return avg_data


if __name__ == "__main__":
    data_loader_train, data_loader_test, num_leagues = get_data_loader()
    train(data_loader_train, data_loader_test, num_leagues)
