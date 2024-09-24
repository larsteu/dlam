import json
import os

import pandas as pd
import torch
from pathlib import Path
import numpy
import sys
from torch.utils.data import DataLoader

from dataset import DatasetWithLeagues, DatasetWithoutLeagues
from models.model import EMModel
from models.model_with_leagues import EMModelWithLeague
from training_step_2 import insert_avg, load_avg_performance_data, COLUMNS_TO_UPDATE, AVERAGE_PERFORMANCE_PATHS, CATEGORICAL_COLUMNS, DROP_COLUMNS
from utils import preprocess_dataset, load_dataset

MODEL_PATH = Path("./trained_models/league_model")
DEVICE = torch.device("cpu")

DATA_PATH = ["./data/evaluation/em24.csv"]

numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    use_base_model = False
    if len(sys.argv) > 1:
        use_base_model = True

    if use_base_model:
        em_model = EMModel()
        MODEL_PATH = Path("./trained_models/base_model")
        model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        em_model.load_state_dict(model['state_dict'])
    else:
        em_model = EMModelWithLeague(36)

        # Load the model
        model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        em_model.load_state_dict(model['model_state_dict'])

    # Set the model to evaluation mode
    em_model.eval()

    # check if the temp file exists
    if os.path.exists("./data/evaluation.csv"):
            data = pd.read_csv("./data/evaluation.csv")
    else:
        # Load the data
        avg_data = load_avg_performance_data(AVERAGE_PERFORMANCE_PATHS)
        data = insert_avg(DATA_PATH, avg_data, COLUMNS_TO_UPDATE)

        # save current dataset to a file named "./data/evaluation.csv"
        data.to_csv("./data/evaluation.csv", index=False)

    # get the columns match_id, home_team and away_team from the data in the same order
    matches = data[["match_nr", "team"]].drop_duplicates().reset_index(drop=True)

    if not use_base_model:
        print("Using the model with leagues")
        # Load the league mapping
        with open("data/league_mapping.json", "r") as f:
            league_mapping = json.load(f)

        # Map league names to IDs
        data['league'] = data['league'].apply(
            lambda x: league_mapping.get(x.replace(".", "").replace(",", "").replace(" ", "").lower(), len(league_mapping)))

        # Preprocess the data
        data = preprocess_dataset(
            data,
            CATEGORICAL_COLUMNS,
            "./data/evaluation/em24mapping.json",
            DROP_COLUMNS,
            remove_player_names=True,
        )


        # Create the dataloader
        dataset = DatasetWithLeagues(data, normalize=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # get the first match and print all players
        num_match = 0
        correct = 0
        incorrect = 0
        for inputs, league_ids, targets in dataloader:
            # get the teams that played in the match from the matches dataframe (the first match is the home team, the second is the away team)
            home_team = matches[matches["match_nr"] == num_match].iloc[0]["team"]
            away_team = matches[matches["match_nr"] == num_match].iloc[1]["team"]

            # Move tensors to the correct device
            inputs = inputs.to(DEVICE)
            league_ids = league_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            # get the prediction from the model
            prediction = em_model(inputs, league_ids)

            prediction = prediction.to("cpu").detach().numpy()

            # check the prediction by comparing the predicted result with the true result
            argmax_pred = numpy.argmax(prediction)
            argmax_target = numpy.argmax(targets)

            if argmax_pred == argmax_target:
                print("correct")
                correct += 1
            else:
                print("incorrect")
                incorrect += 1

            # print the prediction and the teams that played
            #print(f"Match {num_match}: {home_team} vs {away_team} - Prediction: {prediction}")
            #print(f"True Result was {targets}\n---------------------------------------------------------------------\n")

            num_match += 1
    else:
        print("Using the base model")
        # Preprocess the data
        data = preprocess_dataset(
            data,
            CATEGORICAL_COLUMNS,
            "./data/evaluation/em24mapping.json",
            DROP_COLUMNS + ["league"],
            remove_player_names=True,
        )

        # Create the dataloader
        dataset = DatasetWithoutLeagues(data, normalize=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # get the first match and print all players
        num_match = 0
        correct = 0
        incorrect = 0
        for inputs, targets in dataloader:

            # get the teams that played in the match from the matches dataframe (the first match is the home team, the second is the away team)
            home_team = matches[matches["match_nr"] == num_match].iloc[0]["team"]
            away_team = matches[matches["match_nr"] == num_match].iloc[1]["team"]

            # Move tensors to the correct device
            inputs = inputs.to(DEVICE).float()
            targets = targets.to(DEVICE).float()

            # get the prediction from the model
            prediction = em_model(inputs)

            prediction = prediction.to("cpu").detach().numpy()

            # check the prediction by comparing the predicted result with the true result
            argmax_pred = numpy.argmax(prediction)
            argmax_target = numpy.argmax(targets)

            if argmax_pred == argmax_target:
                correct += 1
            else:
                incorrect += 1

            # print the prediction and the teams that played
            print(f"Match {num_match}: {home_team} vs {away_team} - Prediction: {prediction}")
            print(f"True Result was {targets}\n---------------------------------------------------------------------\n")

            num_match += 1


    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy: {correct / (correct + incorrect)}")