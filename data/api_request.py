import json
import pandas as pd
import requests
import sys
import time


# function to write the data to a csv file
def write_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# function to get the player data
def get_player_data(player, match_nr, home_away, game_state, won_loss):

    player_name = player["player"]["name"]

    player_position = player["statistics"][0]["games"]["position"]
    rating = player["statistics"][0]["games"]["rating"]
    minutes_played = player["statistics"][0]["games"]["minutes"]
    attempted_shots = player["statistics"][0]["shots"]["total"]
    shots_on_goal = player["statistics"][0]["shots"]["on"]
    goals = player["statistics"][0]["goals"]["total"]
    assists = player["statistics"][0]["goals"]["assists"]
    saves = player["statistics"][0]["goals"]["saves"]
    conceded_goals = player["statistics"][0]["goals"]["conceded"]
    totat_passes = player["statistics"][0]["passes"]["total"]
    key_passes = player["statistics"][0]["passes"]["key"]
    pass_completion = player["statistics"][0]["passes"]["accuracy"]
    tackles = player["statistics"][0]["tackles"]["total"]
    blocks = player["statistics"][0]["tackles"]["blocks"]
    interceptions = player["statistics"][0]["tackles"]["interceptions"]
    total_duels = player["statistics"][0]["duels"]["total"]
    won_duels = player["statistics"][0]["duels"]["won"]
    attempted_dribbles = player["statistics"][0]["dribbles"]["attempts"]
    successful_dribbles = player["statistics"][0]["dribbles"]["success"]
    cards = (
        player["statistics"][0]["cards"]["yellow"]
        + player["statistics"][0]["cards"]["red"]
    )

    # autofill all the missing values with 0
    if not attempted_shots:
        attempted_shots = 0
    if not shots_on_goal:
        shots_on_goal = 0
    if not goals:
        goals = 0
    if not assists:
        assists = 0
    if not saves:
        saves = 0
    if not conceded_goals:
        conceded_goals = 0
    if not totat_passes:
        totat_passes = 0
    if not key_passes:
        key_passes = 0
    if not pass_completion:
        pass_completion = 0
    if not tackles:
        tackles = 0
    if not blocks:
        blocks = 0
    if not interceptions:
        interceptions = 0
    if not total_duels:
        total_duels = 0
    if not won_duels:
        won_duels = 0
    if not attempted_dribbles:
        attempted_dribbles = 0
    if not successful_dribbles:
        successful_dribbles = 0
    if not cards:
        cards = 0
    if not rating:
        rating = 0
    if not minutes_played:
        minutes_played = 0

    return {
        "match_nr": match_nr,
        "home/away": home_away,
        "player_name": player_name,
        "player_position": player_position,
        "minutes_played": minutes_played,
        "attempted_shots": attempted_shots,
        "shots_on_goal": shots_on_goal,
        "goals": goals,
        "assists": assists,
        "totat_passes": totat_passes,
        "key_passes": key_passes,
        "pass_completion": pass_completion,
        "saves": saves,
        "tackles": tackles,
        "blocks": blocks,
        "interceptions": interceptions,
        "conceded_goals": conceded_goals,
        "total_duels": total_duels,
        "won_duels": won_duels,
        "attempted_dribbles": attempted_dribbles,
        "successful_dribbles": successful_dribbles,
        "cards": cards,
        "game_won": won_loss,
        "game_result": game_state,
        "rating": rating,
    }


if __name__ == "__main__":
    # format: https://v3.football.api-sports.io/fixtures?league=78&season=2010

    # get the console arguments which are [api-key, leagueID, season, filename]
    api_key = sys.argv[1]
    league_id = sys.argv[2]
    seasons = sys.argv[3]
    filename = sys.argv[4]
    match_nr = 0

    start_season = seasons.split("-")[0]
    end_season = seasons.split("-")[1]

    player_data = []

    remaining_requests_per_min = 60
    remaining_requests_day = 1000

    for i in range(int(start_season), int(end_season) + 1):
        current_league_url = (
            "https://v3.football.api-sports.io/fixtures?league="
            + league_id
            + "&season="
            + str(i)
        )

        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io",
        }

        # check if the remaining_requests_per_min is 0 (if yes, wait for 1 minute)
        if remaining_requests_per_min == 0:
            time.sleep(60)

        # check if the remaining_requests_day is 0 (if yes, save the current_data and stop)
        if remaining_requests_day == 0:
            write_to_csv(player_data, "temp.csv")
            break

        try:
            # get the data from the api
            response = requests.request("GET", current_league_url, headers=headers)

            # read into json
            match_data = json.loads(response.text)["response"]

            remaining_requests_per_min = response.headers["X-RateLimit-Remaining"]
            remaining_requests_day = response.headers["x-ratelimit-requests-remaining"]
        except:
            print("Error with season: " + str(i) + "\n")
            continue

        # iterate through the entries and get the ids of the fixtures
        for match in match_data:
            # check if the remaining_requests_per_min is 0 (if yes, wait for 1 minute)
            if remaining_requests_per_min == 0:
                time.sleep(60)

            # check if the remaining_requests_day is 0 (if yes, save the current_data and stop)
            if remaining_requests_day == 0:
                write_to_csv(player_data, "temp.csv")
                break

            fixture = match["fixture"]
            fixture_id = fixture["id"]
            # save the match state
            match_state = (
                str(match["goals"]["home"]) + "-" + str(match["goals"]["away"])
            )
            game_state_home = "won" if match["teams"]["home"]["winner"] else "lost"
            game_state_away = "won" if match["teams"]["away"]["winner"] else "lost"

            if match["teams"]["away"]["winner"] is None:
                game_state_home = "draw"
                game_state_away = "draw"

            try:
                fixture_url = (
                    "https://v3.football.api-sports.io/fixtures/players?fixture="
                    + str(fixture_id)
                )
                response = requests.request("GET", fixture_url, headers=headers)

                # check if the response text is empty (if yes, skip the current match)
                if not response.text:
                    print("No data for match: " + str(fixture_id) + "\n")
                    continue

                home_team = json.loads(response.text)["response"][0]
                away_team = json.loads(response.text)["response"][1]

                number_of_players = 0
                # get the home team players
                for player in home_team["players"]:
                    player_stats = get_player_data(
                        player, match_nr, "home", match_state, game_state_home
                    )

                    # skip all players that have 0 minutes played
                    if player_stats["minutes_played"] == 0:
                        continue

                    player_data.append(player_stats)
                    number_of_players += 1

                # fill up with puffer players until 26 players are reached
                while number_of_players < 26:
                    player_data.append(
                        {
                            "match_nr": match_nr,
                            "home/away": "home",
                            "player_name": "puffer_player",
                            "player_position": "puffer",
                            "minutes_played": 0,
                            "attempted_shots": 0,
                            "shots_on_goal": 0,
                            "goals": 0,
                            "assists": 0,
                            "totat_passes": 0,
                            "key_passes": 0,
                            "pass_completion": 0,
                            "saves": 0,
                            "tackles": 0,
                            "blocks": 0,
                            "interceptions": 0,
                            "conceded_goals": 0,
                            "total_duels": 0,
                            "won_duels": 0,
                            "attempted_dribbles": 0,
                            "successful_dribbles": 0,
                            "cards": 0,
                            "game_won": game_state_home,
                            "game_result": match_state,
                            "rating": 0,
                        }
                    )
                    number_of_players += 1

                # if for some miraculous reason there are more than 26 players, remove the last ones (this should literally never happen, since most leagues can only switch 3 players per game right?)
                while number_of_players > 26:
                    print(
                        "Too many players in home team, removing last player. Somehow\n"
                    )
                    player_data.pop()

                number_of_players = 0

                # get the away team players
                for player in away_team["players"]:
                    player_stats = get_player_data(
                        player, match_nr, "away", match_state, game_state_away
                    )

                    # skip all players that have 0 minutes played
                    if player_stats["minutes_played"] == 0:
                        continue

                    player_data.append(player_stats)
                    number_of_players += 1

                # fill up with puffer players until 26 players are reached
                while number_of_players < 26:
                    player_data.append(
                        {
                            "match_nr": match_nr,
                            "home/away": "away",
                            "player_name": "puffer_player",
                            "player_position": "puffer",
                            "minutes_played": 0,
                            "attempted_shots": 0,
                            "shots_on_goal": 0,
                            "goals": 0,
                            "assists": 0,
                            "totat_passes": 0,
                            "key_passes": 0,
                            "pass_completion": 0,
                            "saves": 0,
                            "tackles": 0,
                            "blocks": 0,
                            "interceptions": 0,
                            "conceded_goals": 0,
                            "total_duels": 0,
                            "won_duels": 0,
                            "attempted_dribbles": 0,
                            "successful_dribbles": 0,
                            "cards": 0,
                            "game_won": game_state_away,
                            "game_result": match_state,
                            "rating": 0,
                        }
                    )
                    number_of_players += 1

                # if for some miraculous reason there are more than 26 players, remove the last ones (this should literally never happen, since most leagues can only switch 3 players per game right?)
                while number_of_players > 26:
                    print(
                        "Too many players in home team, removing last player. Somehow\n"
                    )
                    player_data.pop()

                match_nr += 1
            except:
                print("Error with match: " + str(fixture_id) + "\n")
                continue

            # update the remaining requests per minute and day
            remaining_requests_per_min = response.headers["X-RateLimit-Remaining"]
            remaining_requests_day = response.headers["x-ratelimit-requests-remaining"]
            if match_nr % 50 == 0:
                print("Match number: " + str(match_nr) + " done")
                print(remaining_requests_day + " requests left for today")
                print(remaining_requests_per_min + " requests left for this minute")

    # write the data to a csv file
    try:
        write_to_csv(player_data, filename)
    except:
        # emergency save the data in a text file
        with open("emergency_save.csv", "w") as f:
            for player in player_data:
                f.write(str(player) + "\n")
    print("Data written to player_data.csv")
    print("final match number was: " + str(match_nr) + "\n")
