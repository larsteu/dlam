import pandas as pd
import json
from collections import defaultdict

def get_weights():
    pass
    # 1. Load the trained base model (train_base_model.py must be run before this script)

    # 2. Load the national team matches that we want to train on
    # 2.1 Load the validation dataset (most recent national team matches)
    # For validation, just use the national team matches of this year. If we would want to
    # evaluate on matches from past years we would need to implement a function to retrieve the player
    # statistics of that year.

    # 3. loop with train_epoch_to_determine_weights
    # This loop is similar to train_base_model.py (maybe we need to give the league-info per player here as input)
    # after every epoch: evaluate on the validation set. Now use the average performance of the last season per
    # player as input (like we will do to predict the EM matches)


    # I haven't had a look yet at the dataloader; maybe this is already solved:
    # The only challenge of this script is to get the league of each player.
    # Map each league to a number, you can set the amount of leagues when initializing the model.
    # If we have some leagues with very little players (e.g. only 1 or 2) give it the number num_leagues. This way
    # the league numbers mean the following:
    #   0 to (num_leagues-1): Every number represents a league
    #   num_leagues: "league" that every player is assigned to that is not in one of the num_leagues
    # If we have sufficient players per league, this distinction is not necessary and we do not need this "exta" league.

def create_league_mapping(dataset: pd.DataFrame, output_file):
    league_mapping = {}
    league_players = defaultdict(set)

    next_number = 0

    for index, row in dataset.iterrows():
        league = row["league"]
        player = row['player_name']

        if league not in league_mapping:
            league_mapping[league] = next_number
            next_number += 1

        # if player not puffer player and player not already in league counter add player to league counter
        if player != 'puffer_player':
            league_players[league].add(player)


    league_players = {k: len(v) for k, v in league_players.items()}
    league_players = dict(sorted(league_players.items(), key=lambda item: item[1], reverse=True))

    with open(output_file, 'w') as f:
        json.dump({'mapping': league_mapping, 'players': league_players}, f)


if __name__ == "__main__":
    get_weights()

leagues = {
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
    "Jupiler Pro League": 10,
    "Pro League": 10,
    "1. Division": 11,
    "Ekstraklasa": 12,
    "Premiership": 13,
    "Major League Soccer": 14,
    "Primeira Liga": 15,
    "Sammelkiste": 16
}
