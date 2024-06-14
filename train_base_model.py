import os.path

import torch
from torch.utils.data import DataLoader

from model import EMModel
from dataset import EMDataset
from utils import load_dataset, preprocess_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Model settings ##
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

SAVE_MODEL = True
LOAD_MODEL = True
MODEL_PATH = "data/model_checkpoint"

## Dataset properties ##
TRAIN_DATASET_PATH = [
    "./data/bundesliga_16-23.csv",
    "./data/ligue1_16-23.csv",
    "./data/la_liga_16-23.csv",
    "./data/premier_league_16-23.csv",
    "./data/serie_a_16-23.csv",
]
TEST_DATASET_PATH = ["./data/bundesliga_16-23.csv"]
MAPPINGS_FILE_PATH = "data/mappings_without_names.json"
CATEGORICAL_COLUMNS = ["home/away", "player_name", "player_position"]
DROP_COLUMNS = ["game_won", "rating"]


def train():
    dataset_train = load_dataset(TRAIN_DATASET_PATH)
    dataset_train = preprocess_dataset(
        dataset_train,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH,
        DROP_COLUMNS,
        remove_player_names=True,
    )

    dataset_train = EMDataset(dataset_train, normalize=True)
    data_loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

    dataset_test = load_dataset(TEST_DATASET_PATH)
    dataset_test = preprocess_dataset(
        dataset_test,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH,
        DROP_COLUMNS,
        remove_player_names=True,
    )

    dataset_test = EMDataset(dataset_test, normalize=True)
    data_loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

    em_model = EMModel().to(DEVICE)

    optimizer = torch.optim.Adam(em_model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        em_model.load_model(optimizer, LEARNING_RATE, MODEL_PATH)

    for num_epoch in range(NUM_EPOCHS):
        em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            optimizer=optimizer,
            device=DEVICE,
        )

        em_model.eval_model(dataloader=data_loader_test, device=DEVICE)

        if SAVE_MODEL:
            em_model.save_model(optimizer, MODEL_PATH)


if __name__ == "__main__":
    train()
