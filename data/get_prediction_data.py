import json
import pandas as pd
import requests
import sys
import os
import time
import api_request as api

base_url = "https://v3.football.api-sports.io/"
headers = {"x-rapidapi-key": "", "x-rapidapi-host": "v3.football.api-sports.io"}


def non_null_dict(items):
    result = {}
    for key, value in items:
        if value is None:
            value = 0
        result[key] = value
    return result


def get_page(url):
    # get the data from the api
    response = requests.request("GET", url, headers=headers)

    # read into json
    json_page = json.loads(response.text, object_pairs_hook=non_null_dict)

    data = json_page["response"]
    num_pages = json_page["paging"]["total"]

    remaining_min = response.headers["X-RateLimit-Remaining"]
    remaining_day = response.headers["x-ratelimit-requests-remaining"]
    if int(remaining_min) == 0:
        time.sleep(60)

    if int(remaining_day) == 0:
        print("you hit the daily limit, killing this now")
        sys.exit(1)

    return data, num_pages


def get_league(player_data, original_league_id):

    league = player_data["statistics"][0]["league"]["name"]
    curr_highest = 0
    # iterate over the leagues and get the one with the highest amount of appearances
    for league_data in player_data["statistics"]:
        # skip the original league id
        if league_data["league"]["id"] == original_league_id:
            continue

        if league_data["games"]["appearences"] > curr_highest:
            curr_highest = league_data["games"]["appearences"]
            league = league_data["league"]["name"]

    return league


def get_teams(league, season):
    url = base_url + f"teams?league={league}&season={season}"
    response, num_pages = get_page(url)

    curr_teams = []

    # iterate the response and get the team ids and names
    for curr_team in response:
        curr_teams.append(
            {"team_id": curr_team["team"]["id"], "team_name": curr_team["team"]["name"]}
        )

    if num_pages > 1:
        for i in range(2, num_pages + 1):
            url = base_url + f"teams?league={league}&season={season}&page={i}"
            response, _ = get_page(url)

            for curr_team in response:
                curr_teams.append(
                    {"team_id": curr_team["team"]["id"], "team_name": curr_team["team"]["name"]}
                )

    return curr_teams


def get_players(team, season, league):
    url = base_url + f"players?league={league}&season={season}&team={team}"
    response, num_pages = get_page(url)

    players = []

    # iterate the response and get the player ids and names
    for player in response:

        # check if the player has any appearances
        if player["statistics"][0]["games"]["appearences"] <= 0:
            continue

        players.append(
            {
                "player_id": player["player"]["id"],
                "player_name": player["player"]["name"],
            }
        )

    if num_pages > 1:
        for i in range(2, num_pages + 1):
            url = base_url + f"players?league={league}&season={season}&team={team}&page={i}"
            response, _ = get_page(url)

            for player in response:
                if player["statistics"][0]["games"]["appearences"] <= 0:
                    continue

                players.append(
                    {
                        "player_id": player["player"]["id"],
                        "player_name": player["player"]["name"],
                    }
                )

    return players


def get_player_data(player, season, team, league_id):
    url = base_url + f"players?id={player}&season={season}"

    page, num_pages = get_page(url)

    data = page[0]

    if num_pages > 1:
        print(f"FATAL ERROR ON PLAYER {data['player']['name']}, MORE THAN ONE PAGE")

    # initialize the player data with 0 since it has to be collected from multiple json elements
    player_data = {
        "player_name": data["player"]["name"],
        "player_position": data["statistics"][0]["games"]["position"],
        "minutes_played": 0,
        "attempted_shots": 0,
        "shots_on_goal": 0,
        "goals": 0,
        "assists": 0,
        "total_passes": 0,
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
        "rating": 0,
        "league": get_league(data, league_id),
    }

    # get the total number of games played across all competitions
    total_number_played_games = 0
    for game in data["statistics"]:
        total_number_played_games += game["games"]["appearences"]

    if total_number_played_games == 0:
        return player_data

    # iterate through the competitions and sum up the data, for relative values like the rating the competition is weighted
    for game in data["statistics"]:
        weight = game["games"]["appearences"] / total_number_played_games

        player_data["minutes_played"] += (
            game["games"]["minutes"] / total_number_played_games
        )
        player_data["attempted_shots"] += (
            game["shots"]["total"] / total_number_played_games
        )
        player_data["shots_on_goal"] += game["shots"]["on"] / total_number_played_games
        player_data["goals"] += game["goals"]["total"] / total_number_played_games
        player_data["assists"] += game["goals"]["assists"] / total_number_played_games
        player_data["total_passes"] += (
            game["passes"]["total"] / total_number_played_games
        )
        player_data["key_passes"] += game["passes"]["key"] / total_number_played_games

        # possible edge case where the accuracy actually is 0 over a whole season, but that's gotta be incredibly rare
        if game["passes"]["accuracy"] != 0:
            player_data["pass_completion"] += float(game["passes"]["accuracy"]) * weight
        else:
            player_data["pass_completion"] += player_data["pass_completion"] * weight

        player_data["saves"] += game["goals"]["saves"] / total_number_played_games
        player_data["tackles"] += game["tackles"]["total"] / total_number_played_games
        player_data["blocks"] += game["tackles"]["blocks"] / total_number_played_games
        player_data["interceptions"] += (
            game["tackles"]["interceptions"] / total_number_played_games
        )
        player_data["conceded_goals"] += (
            game["goals"]["conceded"] / total_number_played_games
        )
        player_data["total_duels"] += game["duels"]["total"] / total_number_played_games
        player_data["won_duels"] += game["duels"]["won"] / total_number_played_games
        player_data["attempted_dribbles"] += (
            game["dribbles"]["attempts"] / total_number_played_games
        )
        player_data["successful_dribbles"] += (
            game["dribbles"]["success"] / total_number_played_games
        )
        player_data["cards"] += (
            game["cards"]["yellow"] + game["cards"]["red"]
        ) / total_number_played_games

        # not optimal, but considering mostly competitions with low participation have a null value, this is fine
        if game["games"]["rating"] != 0:
            player_data["rating"] += float(game["games"]["rating"]) * weight
        else:
            player_data["rating"] += player_data["rating"] * weight

    return player_data


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("missing arguments")
        sys.exit(1)

    api_key = sys.argv[1]
    league = sys.argv[2]
    season = sys.argv[3]
    folder = f"{league}_{season}"

    headers["x-rapidapi-key"] = api_key

    # Create the folder
    os.makedirs(folder, exist_ok=True)

    # get the list of participating teams
    teams = get_teams(league, season)

    # use for single teams when errors occured
    '''
    teams = [{
        'team_id': 10,
        'team_name': 'England'
    }]
    '''

    # loop through the teams and get their players
    for team in teams:
        player_data = []

        team_id = team["team_id"]
        team_name = team["team_name"]

        # get the players of the team
        players = get_players(team_id, season, league)

        # set the season -1 because the api uses the starting year, not the ending (i.e. for 23/24 it uses 23)
        player_season = int(season) - 1

        # loop through the players and get their data
        for player in players:
            player_id = player["player_id"]

            try:
                # get the player data
                player_data.append(get_player_data(player_id, player_season, team_name, league))
            except:
                print(f"error with {player['player_name']}")
                continue

        print(f"writing {team_name} to csv")
        # save the player data to a csv
        api.write_to_csv(player_data, f"{folder}/{team_name}.csv")
