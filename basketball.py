"""Train and run a pregame basketball forecaster on JSON Lines game records."""

import argparse
from collections import defaultdict, deque
from datetime import date
from itertools import combinations, groupby
import json
from math import log1p
from pathlib import Path
from context_features import MatchupHistory, pregame_availability


STATS = ("points", "rebounds", "assists")
CAPS = (100, 60, 40)
FEATURES = 82
EXPANDED_FEATURES = 75
RECENT_GAME_FEATURES = 76
PLAYER_SCALES = (30, 12, 10, 36)


class PlayerTotals:
    def __init__(self):
        self.games = 0
        self.sums = [0.0] * len(PLAYER_SCALES)

    def add(self, line):
        self.games += 1
        for index, value in enumerate(line):
            self.sums[index] += value

    def features(self):
        return [*(total / max(self.games, 1) / scale
                  for total, scale in zip(self.sums, PLAYER_SCALES)),
                log1p(self.games) / 8]


class Record:
    def __init__(self):
        self.games = 0
        self.wins = 0

    def add(self, win):
        self.games += 1
        self.wins += win

    def features(self):
        return [self.wins / max(self.games, 1), log1p(self.games) / 8]


class PairHistory:
    def __init__(self):
        self.games = 0
        self.margin = 0

    def add(self, margin):
        self.games += 1
        self.margin += margin


def padded(features, width=FEATURES):
    return features + [0] * (width - len(features))


def read_games(path):
    with open(path, encoding="utf-8") as source:
        games = [json.loads(line) for line in source if line.strip()]
    if len({game["id"] for game in games}) != len(games):
        raise ValueError("Game IDs must be unique.")
    for game in games:
        if date.fromisoformat(game["date"]).isoformat() != game["date"]:
            raise ValueError("Dates must use YYYY-MM-DD.")
    return sorted(games, key=lambda game: (game["date"], game["id"]))


def examples(games, feature_count=FEATURES):
    expanded = feature_count in (EXPANDED_FEATURES, RECENT_GAME_FEATURES)
    use_recent = feature_count == RECENT_GAME_FEATURES
    recent_games = defaultdict(lambda: deque(maxlen=3))
    player_history = defaultdict(lambda: deque(maxlen=10))
    player_season = defaultdict(PlayerTotals)
    player_career = defaultdict(PlayerTotals)
    team_history = defaultdict(lambda: deque(maxlen=10))
    efficiency_history = defaultdict(lambda: deque(maxlen=10))
    team_season = defaultdict(Record)
    coach_career = defaultdict(Record)
    coach_team = defaultdict(Record)
    coach_season = defaultdict(Record)
    last_game = {}
    pairs = defaultdict(PairHistory)
    rotations = defaultdict(lambda: deque(maxlen=10))
    player_team = {}
    matchup = MatchupHistory()
    for _, same_day in groupby(games, key=lambda game: game["date"]):
        updates = []
        for game in same_day:
            day = date.fromisoformat(game["date"])
            season = game.get("season", str(day.year - (day.month < 10)))
            phase = game.get("phase", "unknown")
            if phase not in ("unknown", "regular", "play_in", "playoffs"):
                raise ValueError(f"Invalid phase for {game['id']}.")
            playoff_date = game.get("playoffs_start_as_known")
            days_to_playoffs = max((date.fromisoformat(playoff_date) - day).days, 0) if playoff_date else 0
            scores = game.get("score")
            if scores is not None and scores[0] == scores[1]:
                raise ValueError(f"Final score is tied for {game['id']}.")
            appeared = {}
            availability = {} if expanded else pregame_availability(game)
            current_players = {player: game[side]["team"] for side in ("home", "away")
                               for player in game[side]["roster"]}
            game_rotation = defaultdict(dict)
            if scores is not None:
                for player_id, box in game.get("box", {}).items():
                    if box["minutes"] <= 0:
                        continue
                    target = [box[stat] for stat in STATS]
                    if any(type(value) is not int or not 0 <= value <= cap
                           for value, cap in zip(target, CAPS)):
                        raise ValueError(f"Invalid box score for {player_id} in {game['id']}.")
                    appeared[player_id] = target
                    line = (*target, box["minutes"])
                    volume = (box.get("shots", 0), box.get("free_throws", 0))
                    if box.get("team"):
                        game_rotation[box["team"]][player_id] = (box["minutes"], box["points"])
                        updates.append((player_team.__setitem__, (player_id, box["team"])))
                    updates.extend([
                        (player_history[player_id].append, (line + volume,)),
                        (player_season[player_id, season].add, (line,)),
                        (player_career[player_id].add, (line,)),
                    ])
            tokens, players, targets = [], [], []
            roster_ids = set()
            records = [team_season[game[label]["team"], season] for label in ("home", "away")]
            margins = []
            for label in ("home", "away"):
                prior = team_history[game[label]["team"]]
                margins.append(sum(margin for _, margin in prior) / max(len(prior), 1))
            for side, label in enumerate(("home", "away")):
                team = game[label]
                team_id = team["team"]
                history = team_history[team_id]
                known_rest = team_id in last_game
                rest = min((day - last_game[team_id]).days, 14) if known_rest else 0
                record = team_season[team_id, season]
                form = max(len(history), 1)
                efficiency = efficiency_history[team_id]
                starters = team.get("starters")
                starting_set = set(starters or ())
                lineup_pairs = list(combinations(sorted(starting_set), 2))
                pair_history = {pair: pairs[team_id, pair] for pair in lineup_pairs}
                recent_rotation = rotations[team_id]
                production = defaultdict(lambda: [0., 0.])
                for rotation in recent_rotation:
                    for player, (minutes, points) in rotation.items():
                        production[player][0] += minutes
                        production[player][1] += points
                candidates = sorted(
                    (player for player in production if player not in starting_set
                     and current_players.get(player, player_team.get(player)) == team_id
                     and availability.get((team_id, player), 1.) > 0),
                    key=lambda player: (-production[player][0] * availability.get((team_id, player), 1.), player),
                )[:4] if starters and not expanded and not game.get("history_only") else []
                projected_minutes = {
                    player: production[player][0] / max(len(recent_rotation), 1)
                    * availability.get((team_id, player), 1.)
                    for player in candidates
                }
                unavailable = [player for player in production if availability.get((team_id, player)) == 0]
                missing_minutes, missing_points = 0., 0.
                if starters:
                    usual_five = sorted(production, key=lambda p: (-production[p][0], p))[:5]
                    missing_minutes = sum(production[p][0] for p in usual_five if p not in starting_set)
                    missing_points = sum(production[p][1] for p in usual_five if p not in starting_set)
                tokens.append((f"team/{team_id}", side, padded([
                    sum(win for win, _ in history) / form, rest / 14,
                    len(history) / 10, float(game.get("neutral", False)),
                    record.wins / 82, (record.games - record.wins) / 82,
                    sum(margin for _, margin in history) / form / 30,
                    float(rest == 1), float(known_rest),
                    min(days_to_playoffs, 365) / 365, float(bool(playoff_date)),
                    *(float(phase == name) for name in ("regular", "play_in", "playoffs")),
                    *(sum(row[i] - center for row in efficiency) / max(len(efficiency), 1) / 20
                      for i, center in enumerate((100, 110, 110))),
                    len(efficiency) / 10,
                    missing_minutes / max(len(recent_rotation), 1) / 180,
                    missing_points / max(len(recent_rotation), 1) / 100,
                    len(recent_rotation) / 10,
                    sum(log1p(pair.games) for pair in pair_history.values()) / 70,
                    sum(pair.margin / (pair.games + 10) for pair in pair_history.values()) / 300,
                    *matchup.features(game, day, side)[:52 if expanded else 55],
                    *([] if expanded else [len(candidates) / 4,
                    sum(projected_minutes.values()) / 120,
                    sum(production[p][0] for p in unavailable) / max(len(recent_rotation), 1) / 240,
                    sum((team_id, p) in availability for p in production) / max(len(production), 1)]),
                ], feature_count)))
                coach = team.get("coach")
                coach_records = ([coach_career[coach], coach_team[coach, team_id],
                                  coach_season[coach, season]] if coach else [])
                tokens.append((f"coach/{coach}" if coach else "coach/unknown", side,
                               padded([value for record_stats in coach_records
                                       for value in record_stats.features()], feature_count)))
                if use_recent:
                    for lag, (prior_day, summary) in enumerate(reversed(recent_games[team_id]), 1):
                        tokens.append((f"recent_game/{lag}", side, padded(
                            [*summary, min((day - prior_day).days, 30) / 30], feature_count)))
                roster = team["roster"]
                if starters is not None and (len(starters) != 5 or len(set(starters)) != 5
                                             or not set(starters).issubset(roster)):
                    raise ValueError(f"Expected five unique {label} starters on the roster for {game['id']}.")
                for player_id in sorted(set(candidates) - set(roster)) + sorted(roster):
                    if player_id in roster_ids:
                        raise ValueError(f"Duplicate roster player {player_id} in {game['id']}.")
                    roster_ids.add(player_id)
                    prior = player_history[player_id]
                    means = [sum(row[i] for row in prior) / max(len(prior), 1) for i in range(4)]
                    teammates = [entry for pair, entry in pair_history.items() if player_id in pair]
                    players.append((player_id, side, padded([
                        *(value / scale for value, scale in zip(means, PLAYER_SCALES)),
                        len(prior) / 10,
                        0 if starters is None else (1 if player_id in starters else -1),
                        *player_season[player_id, season].features(),
                        *player_career[player_id].features(),
                        *(sum(row[i] for row in prior) / max(len(prior), 1) / scale
                          for i, scale in ((4, 20), (5, 10))),
                        *([log1p(entry.games) / 7 for entry in sorted(
                            teammates, key=lambda entry: (entry.games, entry.margin), reverse=True)]
                          + ([] if expanded else [0.] * (4 - len(teammates)))),
                        sum(entry.margin / (entry.games + 10) for entry in teammates) / 120,
                        *([] if expanded else [projected_minutes.get(player_id, 0.) / 36,
                        availability.get((team_id, player_id), 0.),
                        float((team_id, player_id) in availability)]),
                    ], feature_count)))
                    targets.append(appeared.get(player_id))
                if scores is not None:
                    margin = scores[side] - scores[1 - side]
                    win = int(margin > 0)
                    updates.append((history.append, ((win, margin),)))
                    if use_recent:
                        opponent = records[1 - side]
                        summary = [
                            margin / 25, float(win), (scores[side] - 110) / 25,
                            (scores[1 - side] - 110) / 25, *opponent.features(),
                            float(side == 0 and not game.get("neutral")), rest / 14,
                            missing_minutes / max(len(recent_rotation), 1) / 180,
                            float(starters is not None),
                        ]
                        updates.append((recent_games[team_id].append, ((day, summary),)))
                    team_box = game.get("team_box")
                    if team_box and all(value is not None for value in team_box[side]):
                        updates.append((efficiency.append, (team_box[side],)))
                    updates.append((last_game.__setitem__, (team_id, day)))
                    if game_rotation[team_id]:
                        updates.append((recent_rotation.append, (game_rotation[team_id],)))
                    updates.extend((entry.add, (margin,)) for entry in pair_history.values())
                    updates.extend((entry.add, (win,)) for entry in [record, *coach_records])
            win = int(scores[0] > scores[1]) if scores is not None and not game.get("history_only") else None
            if scores is not None:
                margin = scores[0] - scores[1]
                updates.append((matchup.update, (game, day, [margin + margins[1], -margin + margins[0]])))
            yield game, tokens, players, win, targets
        for update, values in updates:
            update(*values)


def split(rows, validation_start, test_start, training_start=None, test_end=None):
    boundaries = [date.fromisoformat(value) for value in (validation_start, test_start)]
    if [day.isoformat() for day in boundaries] != [validation_start, test_start]:
        raise ValueError("Split dates must use YYYY-MM-DD.")
    if boundaries[0] >= boundaries[1]:
        raise ValueError("Validation must start before test.")
    if test_end is not None and test_end <= boundaries[1]:
        raise ValueError("Test end must follow test start.")
    if training_start is not None and training_start >= boundaries[0]:
        raise ValueError("Training must start before validation.")
    train, validation, test = [], [], []
    for row in rows:
        day = row[0]["date"]
        if test_end is not None and day >= test_end.isoformat():
            break
        if training_start is not None and day < training_start.isoformat():
            continue
        destination = train if day < validation_start else validation if day < test_start else test
        if row[3] is not None:
            destination.append(row)
    if not all((train, validation, test)):
        raise ValueError("Each chronological split needs completed games.")
    return train, validation, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("games")
    train.add_argument("checkpoint", type=Path)
    train.add_argument("--validation-start")
    train.add_argument("--training-start", type=date.fromisoformat)
    train.add_argument("--test-start")
    train.add_argument("--test-end", type=date.fromisoformat)
    train.add_argument("--refit-all", action="store_true")
    train.add_argument("--expanded", action="store_true")
    train.add_argument("--recent-games", action="store_true",
                       help="Use expanded features plus tokens for each team's last three games.")
    train.add_argument("--epochs", default=15, type=int)
    train.add_argument("--width", default=64, choices=(64, 128, 256), type=int)
    train.add_argument("--token-layers", default=0, choices=range(4), type=int,
                       help="Number of extra learned token transformations before attention.")
    train.add_argument("--attention-layers", default=2, choices=(2, 3), type=int)
    train.add_argument("--bidirectional", action="store_true")
    train.add_argument("--team-comparison", action="store_true")
    train.add_argument("--margin-loss", action="store_true")
    train.add_argument("--margin-weight", type=float)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--batch-size", default=32, type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("games")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--test-start", required=True, type=date.fromisoformat)
    evaluate.add_argument("--test-end", type=date.fromisoformat)
    evaluate.add_argument("--ensemble", nargs="+", type=Path, default=[])
    predict = commands.add_parser("predict")
    predict.add_argument("games")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("--game-id", required=True)
    predict.add_argument("--ensemble", nargs="+", type=Path, default=[])
    predict.add_argument("--prop", action="append", nargs=3, default=[],
                         metavar=("PLAYER", "STAT", "LINE"))
    args = parser.parse_args()

    from transformer import evaluate_checkpoint, fit, forecast, load_checkpoint

    games = read_games(args.games)
    if args.command == "train":
        feature_count = EXPANDED_FEATURES if args.expanded else FEATURES
        if args.recent_games:
            feature_count = RECENT_GAME_FEATURES
        rows = examples(games, feature_count)
        if args.refit_all:
            if args.validation_start or args.test_start or args.training_start or args.test_end:
                parser.error("--refit-all uses every eligible game in the file.")
            splits = ([row for row in rows if row[3] is not None], [], [])
        else:
            if not args.validation_start or not args.test_start:
                parser.error("Training requires validation and test dates or --refit-all.")
            splits = split(rows, args.validation_start, args.test_start, args.training_start, args.test_end)
        print(json.dumps(fit(splits, args), indent=2))
        return
    checkpoint = load_checkpoint(args.checkpoint)
    rows = examples(games, checkpoint[0].features.in_features)
    if args.command == "evaluate":
        print(json.dumps(evaluate_checkpoint(rows, args, checkpoint), indent=2))
        return
    for row in rows:
        if row[0]["id"] == args.game_id:
            print(json.dumps(forecast(row, args, checkpoint), indent=2))
            return
    parser.error(f"Game ID {args.game_id!r} was not found.")


if __name__ == "__main__":
    main()

