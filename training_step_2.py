import os.path
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
MODEL_PATH = Path("./trained_models/leuage_model")

INITIAL_MODEL_PATH = Path("./trained_models/base_model")  # Path to the initial EMModel checkpoint

## Dataset properties ##
TRAIN_DATASET_PATH = ["./data/nations_league.csv"]
TEST_DATASET_PATH = ["./data/nations_league.csv"]
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

    # 2. Test data
    dataset_test = load_dataset(TEST_DATASET_PATH)
    dataset_test = preprocess_dataset(
        dataset_test,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TEST,
        DROP_COLUMNS,
        remove_player_names=True,
    )
    dataset_test = DatasetWithLeagues(dataset_test, normalize=True, use_existing_normalisation=True)
    data_loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

    # 3. Number of leagues & return
    num_leagues = len(dataset_train.league_mapping)

    return data_loader_train, data_loader_test, num_leagues


def train(data_loader_train, data_loader_test, num_leagues):
    em_model = EMModelWithLeague(num_leagues=num_leagues).to(DEVICE)

    # Load the base model
    assert os.path.exists(INITIAL_MODEL_PATH), f"Initial model path {INITIAL_MODEL_PATH} does not exist. Please run training_step1.py first."  # fmt: skip
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
        best_eval_loss = em_model.eval_model(dataloader=data_loader_test, device=DEVICE)

    best_eval_loss = None

    for num_epoch in range(NUM_EPOCHS):
        em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            loss_fn=em_model.get_loss(),
            optimizer=optimizer,
            device=DEVICE,
        )

        curr_eval_loss = em_model.eval_model(dataloader=data_loader_test, device=DEVICE)

        if best_eval_loss is None or curr_eval_loss < best_eval_loss:
            em_model.save_model(optimizer, MODEL_PATH)
            best_eval_loss = curr_eval_loss


if __name__ == "__main__":
    data_loader_train, data_loader_test, num_leagues = get_data_loader()
    train(data_loader_train, data_loader_test, num_leagues)
