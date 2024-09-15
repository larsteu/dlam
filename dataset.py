import pandas as pd
from torch.utils.data import Dataset
from utils import normalize_dataset
import numpy as np


class DatasetWithoutLeagues(Dataset):
    def __init__(self, dataset: pd.DataFrame, normalize=False, use_existing_normalisation=False):
        self.dataset = dataset
        self.normalize = normalize
        self.dataset["team_1_goals"] = self.dataset["game_result"].map(lambda x: int(x.split(sep="-")[0]))
        self.dataset["team_2_goals"] = self.dataset["game_result"].map(lambda x: int(x.split(sep="-")[1]))
        self.labels = self._create_labels()
        self.dataset = dataset.drop(columns="game_result")
        if self.normalize:
            self.dataset = normalize_dataset(
                self.dataset,
                "data/normalization_info.json",
                use_existing_normalisation=use_existing_normalisation,
            )

        self.data = self.dataset.drop(columns=["team_1_goals", "team_2_goals"]).values
        self.data = np.array([[self.data[i : i + 52]] for i in range(0, len(self.data), 52)])

    def _create_labels(self):
        team_1_goals = self.dataset["team_1_goals"].values
        team_2_goals = self.dataset["team_2_goals"].values
        # create labels with size (n, 3) where n is the number of games (n = len(team_1_goals) / 52)
        n: int = len(team_1_goals) // 52
        labels: [float] = np.zeros((n, 3))
        curr_match = 0
        for i in range(0, len(team_1_goals), 52):
            if team_1_goals[i] > team_2_goals[i]:
                labels[curr_match] = [1, 0, 0]
            elif team_1_goals[i] < team_2_goals[i]:
                labels[curr_match] = [0, 0, 1]
            else:
                labels[curr_match] = [0, 1, 0]
            curr_match += 1
        return labels

    def __len__(self):
        return int(len(self.dataset) / 52)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DatasetWithLeagues(DatasetWithoutLeagues):

    def __init__(self, dataset: pd.DataFrame, normalize=False, use_existing_normalisation=False):
        # We store the leagues separately and remove them from the dataset
        leagues = dataset["league"]
        dataset = dataset.drop(columns="league")

        # Call the parent constructor (i.e. of DatasetWithoutLeagues)
        super().__init__(dataset, normalize, use_existing_normalisation)

        # Create the league data array
        self.league_data = np.array([leagues.iloc[i:i+52].values for i in range(0, len(leagues), 52)])


    def __getitem__(self, idx):
        # Get data and labels as we would normally from the DatasetWithoutLeagues class
        data, labels = super().__getitem__(idx)

        # Get the league data for this batch
        league = self.league_data[idx]

        # Return the data, league, and labels
        return data, league, labels


# All leagues: {"mapping": {"Premier League": 0, "Bundesliga": 1, "La Liga": 2, "Ligue 1": 3, "Serie A": 4, "World Cup": 5, "puffer": 6, "Liga Profesional Argentina": 7, "World Cup - Qualification South America": 8, "Eredivisie": 9, "Liga MX": 10, "S\u00fcper Lig": 11, "Primera Divisi\u00f3n - Clausura": 12, "Primeira Liga": 13, "J1 League": 14, "Championship": 15, "UEFA Europa League": 16, "Segunda Divisi\u00f3n": 17, "Super League": 18, "Jupiler Pro League": 19, "World Cup - Qualification CONCACAF": 20, "Major League Soccer": 21, "World Cup - Qualification Asia": 22, "Friendlies": 23, "Premiership": 24, "Superettan": 25, "Pro League": 26, "World Cup - Qualification Europe": 27, "CONMEBOL Libertadores": 28, "Allsvenskan": 29, "Super League 1": 30, "Stars League": 31, "Persian Gulf Pro League": 32, "A-League": 33, "2. Bundesliga": 34, "\u00darvalsdeild": 35, "1. Lig": 36, "Superliga": 37, "Serie B": 38, "I-League": 39, "Primera Divisi\u00f3n": 40, "K League 1": 41, "Liga I": 42, "Super Liga": 43, "Primera A": 44, "Ligat Ha'al": 45, "Tercera Divisi\u00f3n RFEF": 46, "National 2 - Group A": 47, "CAF Champions League": 48, "Primera Divisi\u00f3n RFEF": 49, "League One": 50, "K League 2": 51, "Liga Paname\u00f1a de F\u00fatbol": 52, "First League": 53, "AFC Champions League": 54, "HNL": 55, "Ekstraklasa": 56, "World Cup - Qualification Africa": 57, "1. Division": 58, "AFC U23 Asian Cup": 59, "Ligue 2": 60, "Challenger Pro League": 61, "UEFA Nations League": 62}, "players": {"Premier League": 210, "La Liga": 103, "Bundesliga": 96, "Serie A": 89, "Ligue 1": 82, "Pro League": 45, "Major League Soccer": 34, "S\u00fcper Lig": 32, "Liga MX": 26, "Championship": 25, "Eredivisie": 23, "Primeira Liga": 19, "Primera Divisi\u00f3n": 19, "World Cup - Qualification Asia": 17, "Jupiler Pro League": 16, "World Cup - Qualification South America": 14, "Super League": 13, "J1 League": 11, "World Cup - Qualification CONCACAF": 10, "Premiership": 10, "Stars League": 10, "Friendlies": 9, "Super League 1": 9, "Persian Gulf Pro League": 9, "Liga Profesional Argentina": 8, "2. Bundesliga": 8, "K League 1": 8, "World Cup - Qualification Europe": 6, "HNL": 6, "World Cup": 5, "Segunda Divisi\u00f3n": 5, "Superliga": 5, "Primera Divisi\u00f3n - Clausura": 4, "Ekstraklasa": 4, "Allsvenskan": 3, "Serie B": 3, "Primera A": 3, "Super Liga": 2, "CAF Champions League": 2, "League One": 2, "K League 2": 2, "Liga Paname\u00f1a de F\u00fatbol": 2, "1. Division": 2, "UEFA Nations League": 2, "UEFA Europa League": 1, "Superettan": 1, "CONMEBOL Libertadores": 1, "A-League": 1, "\u00darvalsdeild": 1, "1. Lig": 1, "I-League": 1, "Liga I": 1, "Ligat Ha'al": 1, "Tercera Divisi\u00f3n RFEF": 1, "National 2 - Group A": 1, "Primera Divisi\u00f3n RFEF": 1, "First League": 1, "AFC Champions League": 1, "World Cup - Qualification Africa": 1, "AFC U23 Asian Cup": 1, "Ligue 2": 1, "Challenger Pro League": 1}}
