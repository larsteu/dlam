import torch

def get_weights():
    # 1. Load the trained base model (train_base_model.py must be run before this script)
    # TODO: implement this

    # 2. Load the national team matches that we want to train on
    # 2.1 Load the validation dataset (most recent national team matches)
    # For validation, just use the national team matches of this year. If we would want to evaluate on matches from past years we would need to implement a function to retrieve the player statistics of that year.
    # TODO: implement this

    # 3. loop with train_epoch_to_determine_weights
    # This loop is similar to train_base_model.py (maybe we need to give the league-info per player here as input)
    # after every epoch: evaluate on the validation set. Now use the average performance of the last season per player as input (like we will do to predict the EM matches)
    # TODO: implement this

    # I haven't had a look yet at the dataloader; maybe this is already solved:
    # The only challenge of this script is to get the league of each player.
    # Map each league to a number, you can set the amount of leagues when initializing the model.
    # If we have some leagues with very little players (e.g. only 1 or 2) give it the number num_leagues. This way the league numbers mean the following:
    #   0 to (num_leagues-1): Every number represents a league
    #   num_leagues: "league" that every player is assigned to that is not in one of the num_leagues
    # If we have sufficient players per league, this distinction is not necessary and we do not need this "exta" league.

if __name__ == '__main__':
    get_weights()
