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
        self.dataset = dataset.drop(columns="game_result")
        if self.normalize:
            self.dataset = normalize_dataset(
                self.dataset,
                "data/normalization_info.json",
                use_existing_normalisation=use_existing_normalisation,
            )

    def __len__(self):
        return int(len(self.dataset) / 52)

    def __getitem__(self, idx):
        label_1 = self.dataset.iloc[idx * 52].values[-2]
        label_2 = self.dataset.iloc[idx * 52].values[-1]
        data = self.dataset.drop(columns=["team_1_goals", "team_2_goals"])
        data = data.iloc[idx * 52 : (idx * 52) + 52].values
        return np.array([data]), np.array([label_1, label_2])


class DatasetWithLeagues(DatasetWithoutLeagues):
    league_mapping = {
        "Bundesliga": 0,
        "Premier League": 1,
        "Serie A": 2,
        "La Liga": 3,
        "Primera División": 3,
        "Ligue 1": 4,
        "Eredivisie": 5,
        "Süper Lig": 6,
        "Championship": 7,
        "Super League": 8,
        "Czech Liga": 9,
        "Fortuna Liga": 9,
        "Jupiler Pro League": 10,
        "Pro League": 10,
        "1. Division": 11,
        "Ekstraklasa": 12,
        "Premiership": 13,
        "Major League Soccer": 14,
        "Primeira Liga": 15,
        "Sammelkiste": 16,
    }

    def __init__(self, dataset: pd.DataFrame, normalize=False, use_existing_normalisation=False):
        # We store the leagues separately and remove them from the dataset
        self.leagues: pd.DataFrame = dataset["league"]
        dataset = dataset.drop(columns="league")

        # Call the parent constructor (i.e. of DatasetWithoutLeagues)
        super().__init__(dataset, normalize, use_existing_normalisation)

        # Process the league names: map them to integers
        self.leagues = self.leagues.map(lambda x: self.league_mapping.get(x, 16))

    def __getitem__(self, idx):
        # Get data as we would normally from the DatasetWithoutLeagues class
        data, labels = super().__getitem__(idx)

        # Get the league for this batch
        league = self.leagues.iloc[idx * 52 : (idx * 52) + 52].values

        # Return the data, league, and labels
        return data, league, labels


# All leagues: {"mapping": {"Premier League": 0, "Bundesliga": 1, "La Liga": 2, "Ligue 1": 3, "Serie A": 4, "World Cup": 5, "puffer": 6, "Liga Profesional Argentina": 7, "World Cup - Qualification South America": 8, "Eredivisie": 9, "Liga MX": 10, "S\u00fcper Lig": 11, "Primera Divisi\u00f3n - Clausura": 12, "Primeira Liga": 13, "J1 League": 14, "Championship": 15, "UEFA Europa League": 16, "Segunda Divisi\u00f3n": 17, "Super League": 18, "Jupiler Pro League": 19, "World Cup - Qualification CONCACAF": 20, "Major League Soccer": 21, "World Cup - Qualification Asia": 22, "Friendlies": 23, "Premiership": 24, "Superettan": 25, "Pro League": 26, "World Cup - Qualification Europe": 27, "CONMEBOL Libertadores": 28, "Allsvenskan": 29, "Super League 1": 30, "Stars League": 31, "Persian Gulf Pro League": 32, "A-League": 33, "2. Bundesliga": 34, "\u00darvalsdeild": 35, "1. Lig": 36, "Superliga": 37, "Serie B": 38, "I-League": 39, "Primera Divisi\u00f3n": 40, "K League 1": 41, "Liga I": 42, "Super Liga": 43, "Primera A": 44, "Ligat Ha'al": 45, "Tercera Divisi\u00f3n RFEF": 46, "National 2 - Group A": 47, "CAF Champions League": 48, "Primera Divisi\u00f3n RFEF": 49, "League One": 50, "K League 2": 51, "Liga Paname\u00f1a de F\u00fatbol": 52, "First League": 53, "AFC Champions League": 54, "HNL": 55, "Ekstraklasa": 56, "World Cup - Qualification Africa": 57, "1. Division": 58, "AFC U23 Asian Cup": 59, "Ligue 2": 60, "Challenger Pro League": 61, "UEFA Nations League": 62}, "players": {"Premier League": 210, "La Liga": 103, "Bundesliga": 96, "Serie A": 89, "Ligue 1": 82, "Pro League": 45, "Major League Soccer": 34, "S\u00fcper Lig": 32, "Liga MX": 26, "Championship": 25, "Eredivisie": 23, "Primeira Liga": 19, "Primera Divisi\u00f3n": 19, "World Cup - Qualification Asia": 17, "Jupiler Pro League": 16, "World Cup - Qualification South America": 14, "Super League": 13, "J1 League": 11, "World Cup - Qualification CONCACAF": 10, "Premiership": 10, "Stars League": 10, "Friendlies": 9, "Super League 1": 9, "Persian Gulf Pro League": 9, "Liga Profesional Argentina": 8, "2. Bundesliga": 8, "K League 1": 8, "World Cup - Qualification Europe": 6, "HNL": 6, "World Cup": 5, "Segunda Divisi\u00f3n": 5, "Superliga": 5, "Primera Divisi\u00f3n - Clausura": 4, "Ekstraklasa": 4, "Allsvenskan": 3, "Serie B": 3, "Primera A": 3, "Super Liga": 2, "CAF Champions League": 2, "League One": 2, "K League 2": 2, "Liga Paname\u00f1a de F\u00fatbol": 2, "1. Division": 2, "UEFA Nations League": 2, "UEFA Europa League": 1, "Superettan": 1, "CONMEBOL Libertadores": 1, "A-League": 1, "\u00darvalsdeild": 1, "1. Lig": 1, "I-League": 1, "Liga I": 1, "Ligat Ha'al": 1, "Tercera Divisi\u00f3n RFEF": 1, "National 2 - Group A": 1, "Primera Divisi\u00f3n RFEF": 1, "First League": 1, "AFC Champions League": 1, "World Cup - Qualification Africa": 1, "AFC U23 Asian Cup": 1, "Ligue 2": 1, "Challenger Pro League": 1}}
