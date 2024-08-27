import json
import pandas as pd
import requests
import sys
import os
import time
from tqdm import tqdm

base_url = "https://v3.football.api-sports.io/"
headers = {"x-rapidapi-key": "", "x-rapidapi-host": "v3.football.api-sports.io"}
all_data = []

def non_null_dict(items):
    return {key: value if value is not None else 0 for key, value in items}


def get_page(url):
    response = requests.request("GET", url, headers=headers)
    json_page = json.loads(response.text, object_pairs_hook=non_null_dict)
    data = json_page["response"]
    num_pages = json_page["paging"]["total"]

    remaining_min = int(response.headers["X-RateLimit-Remaining"])
    remaining_day = int(response.headers["x-ratelimit-requests-remaining"])

    if remaining_min == 0:
        time.sleep(60)
    if remaining_day == 0:
        print("You hit the daily limit, exiting now")
        # save the current state to a file named temp.csv
        df = pd.DataFrame(all_data)
        df.to_csv("temp.csv", index=False)
        sys.exit(1)

    return data, num_pages


def get_fixtures(league_id, season):
    url = f"{base_url}fixtures?league={league_id}&season={season}"
    fixtures, num_pages = get_page(url)

    all_fixtures = fixtures
    if num_pages > 1:
        for i in range(2, num_pages + 1):
            url = f"{base_url}fixtures?league={league_id}&season={season}&page={i}"
            fixtures, _ = get_page(url)
            all_fixtures.extend(fixtures)

    return all_fixtures


def get_lineup(fixture_id):
    url = f"{base_url}fixtures/lineups?fixture={fixture_id}"
    lineups, _ = get_page(url)
    return lineups


def get_player_season_stats(player_id, team_id, league_id, season, stats_buffer):
    # Check if player stats are already in the buffer
    if player_id in stats_buffer:
        return stats_buffer[player_id]

    url = f"{base_url}players?id={player_id}&season={season}&league={league_id}&team={team_id}"
    players, _ = get_page(url)

    if players and players[0]['statistics']:
        stats = players[0]['statistics'][0]
        # Store the stats in the buffer
        stats_buffer[player_id] = stats
        return stats

    return None


def format_player_stats(player, stats, is_home, game_result):
    home_away = "home" if is_home else "away"
    game_won = "won" if (is_home and game_result[0] > game_result[2]) or \
                        (not is_home and game_result[2] > game_result[0]) else \
        "lost" if (is_home and game_result[0] < game_result[2]) or \
                  (not is_home and game_result[2] < game_result[0]) else "draw"

    return {
        "home/away": home_away,
        "player_name": player['player']['name'],
        "player_position": stats['games']['position'] if stats['games']['position'] else "-",
        "minutes_played": stats['games']['minutes'] // stats['games']['appearences'] if stats['games'][
                                                                                            'appearences'] > 0 else 0,
        "attempted_shots": stats['shots']['total'] // stats['games']['appearences'] if stats['games'][
                                                                                           'appearences'] > 0 else 0,
        "shots_on_goal": stats['shots']['on'] // stats['games']['appearences'] if stats['games'][
                                                                                      'appearences'] > 0 else 0,
        "goals": stats['goals']['total'] // stats['games']['appearences'] if stats['games']['appearences'] > 0 else 0,
        "assists": stats['goals']['assists'] // stats['games']['appearences'] if stats['games'][
                                                                                     'appearences'] > 0 else 0,
        "total_passes": stats['passes']['total'] // stats['games']['appearences'] if stats['games'][
                                                                                         'appearences'] > 0 else 0,
        "key_passes": stats['passes']['key'] // stats['games']['appearences'] if stats['games'][
                                                                                     'appearences'] > 0 else 0,
        "pass_completion": f"{stats['passes']['accuracy']}%" if stats['passes']['accuracy'] is not None else "0%",
        "saves": stats['goals']['saves'] // stats['games']['appearences'] if stats['games']['appearences'] > 0 else 0,
        "tackles": stats['tackles']['total'] // stats['games']['appearences'] if stats['games'][
                                                                                     'appearences'] > 0 else 0,
        "blocks": stats['tackles']['blocks'] // stats['games']['appearences'] if stats['games'][
                                                                                     'appearences'] > 0 else 0,
        "interceptions": stats['tackles']['interceptions'] // stats['games']['appearences'] if stats['games'][
                                                                                                   'appearences'] > 0 else 0,
        "conceded_goals": stats['goals']['conceded'] // stats['games']['appearences'] if stats['games'][
                                                                                             'appearences'] > 0 else 0,
        "total_duels": stats['duels']['total'] // stats['games']['appearences'] if stats['games'][
                                                                                       'appearences'] > 0 else 0,
        "won_duels": stats['duels']['won'] // stats['games']['appearences'] if stats['games']['appearences'] > 0 else 0,
        "attempted_dribbles": stats['dribbles']['attempts'] // stats['games']['appearences'] if stats['games'][
                                                                                                    'appearences'] > 0 else 0,
        "successful_dribbles": stats['dribbles']['success'] // stats['games']['appearences'] if stats['games'][
                                                                                                    'appearences'] > 0 else 0,
        "cards": (stats['cards']['yellow'] + stats['cards']['red']) // stats['games']['appearences'] if stats['games'][
                                                                                                            'appearences'] > 0 else 0,
        "game_won": game_won,
        "game_result": game_result,
        "rating": stats['games']['rating'] if stats['games']['rating'] is not None else "-"
    }


def create_puffer_player(home_away, game_won, game_result):
    return {
        "home/away": home_away,
        "player_name": "puffer_player",
        "player_position": "puffer",
        "minutes_played": 0,
        "attempted_shots": 0,
        "shots_on_goal": 0,
        "goals": 0,
        "assists": 0,
        "total_passes": 0,
        "key_passes": 0,
        "pass_completion": "0%",
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
        "game_won": game_won,
        "game_result": game_result,
        "rating": 0
    }


def main(api_key, league_id, start_season, end_season, output_file):
    headers["x-rapidapi-key"] = api_key

    for season in range(start_season, end_season + 1):
        print(f"Processing season {season}")
        stats_buffer = {}  # Reset stats buffer for each season
        fixtures = get_fixtures(league_id, season)
        loop = tqdm(fixtures)

        for match_nr, fixture in enumerate(loop):

            lineups = get_lineup(fixture["fixture"]["id"])
            match_data = []
            game_result = f"{fixture['goals']['home']}-{fixture['goals']['away']}"

            for team in lineups:
                is_home = team["team"]["id"] == fixture["teams"]["home"]["id"]
                home_away = "home" if is_home else "away"
                game_won = "won" if (is_home and fixture['goals']['home'] > fixture['goals']['away']) or \
                                    (not is_home and fixture['goals']['away'] > fixture['goals']['home']) else \
                    "lost" if (is_home and fixture['goals']['home'] < fixture['goals']['away']) or \
                              (not is_home and fixture['goals']['away'] < fixture['goals']['home']) else "draw"

                team_players = []
                for player in team["startXI"] + team["substitutes"]:
                    stats = get_player_season_stats(player["player"]["id"], team["team"]["id"], league_id, season,
                                                    stats_buffer)
                    if stats:
                        player_data = format_player_stats(player, stats, is_home, game_result)
                        player_data["match_nr"] = match_nr
                        team_players.append(player_data)

                # Add puffer players if needed
                while len(team_players) < 26:
                    puffer_player = create_puffer_player(home_away, game_won, game_result)
                    puffer_player["match_nr"] = match_nr
                    team_players.append(puffer_player)

                match_data.extend(team_players)

            all_data.extend(match_data)

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Data written to {output_file}")
    print(f"Total unique players processed: {len(stats_buffer)}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py <api_key> <league_id> <start_season> <end_season> <output_file>")
        sys.exit(1)

    api_key = sys.argv[1]
    league_id = int(sys.argv[2])
    start_season = int(sys.argv[3])
    end_season = int(sys.argv[4])
    output_file = sys.argv[5]

    main(api_key, league_id, start_season, end_season, output_file)