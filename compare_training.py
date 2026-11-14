"""Run matched chronological winner-model comparisons."""

from collections import defaultdict, deque
from itertools import groupby
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from basketball import EXPANDED_FEATURES, examples, read_games, split
from transformer import batch, load_checkpoint


def score(truth, logits):
    labels = torch.tensor(truth, dtype=torch.float32)
    predictions = torch.as_tensor(logits, dtype=torch.float32)
    correct = int(((predictions >= 0) == labels.bool()).sum())
    return dict(games=len(truth), correct=correct, accuracy=correct / len(truth),
                log_loss=F.binary_cross_entropy_with_logits(predictions, labels).item())


def adaptive_predictions(checkpoint, replay, test, interval, learning_rate):
    torch.manual_seed(17)
    randomizer = random.Random(17)
    model, metadata = load_checkpoint(checkpoint)
    device = torch.device("cpu")
    vocab = metadata["vocab"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pending, predictions = [], []
    steps = 0
    for _, rows in groupby(test, key=lambda row: row[0]["date"]):
        day = list(rows)
        model.eval()
        with torch.no_grad():
            inputs = batch(day, vocab, device, winner_only=True)
            predictions.extend(model(*inputs[:6], winner_only=True).tolist())
        if not interval:
            continue
        pending.extend(day)
        while len(pending) >= interval:
            rows = pending[:interval] + randomizer.sample(replay, interval)
            del pending[:interval]
            model.train()
            inputs = batch(rows, vocab, device, winner_only=True)
            winner, margin = model(*inputs[:6], winner_only=True, return_margin=True)
            actual = margin.new_tensor([(row[0]["score"][0] - row[0]["score"][1]) / 15
                                        for row in rows])
            loss = F.binary_cross_entropy_with_logits(winner, inputs[-2])
            loss = loss + .25 * F.smooth_l1_loss(margin, actual)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            steps += 1
    return predictions, steps


def adjusted_efficiency(games):
    raw = defaultdict(lambda: deque(maxlen=20))
    adjusted = defaultdict(lambda: deque(maxlen=20))
    for _, day in groupby(games, key=lambda game: game["date"]):
        updates = []
        for game in day:
            teams = [game[side]["team"] for side in ("home", "away")]
            features = []
            for team in teams:
                history = adjusted[team]
                for window in (10, 20):
                    recent = list(history)[-window:]
                    features.extend([sum(row[i] for row in recent) / max(len(recent), 1)
                                     for i in (0, 1)])
                    features.append(len(recent))
            yield game["id"], features
            team_box = game.get("team_box")
            if game.get("score") is None or not team_box:
                continue
            for i, team in enumerate(teams):
                _, offense, defense = team_box[i]
                opponent = raw[teams[1 - i]]
                if offense is None or defense is None:
                    continue
                if opponent:
                    expected_offense = sum(row[1] for row in opponent) / len(opponent)
                    expected_defense = sum(row[0] for row in opponent) / len(opponent)
                    updates.append((adjusted[team], (offense - expected_offense,
                                                    expected_defense - defense)))
                updates.append((raw[team], (offense, defense)))
        for history, entry in updates:
            history.append(entry)


def tabular(row):
    import numpy as np

    sides = []
    for side in (0, 1):
        team = next(features for name, token_side, features in row[1]
                    if token_side == side and name.startswith("team/"))
        players = np.asarray([features[:23] for _, token_side, features in row[2]
                              if token_side == side], dtype=float).reshape(-1, 23)
        summaries = np.r_[players.mean(0), players.max(0), players.min(0)] if len(players) else np.zeros(69)
        sides.append(np.r_[team, summaries])
    home, away = sides
    return np.r_[home, away, home - away]


def fit_margin_tree(train_x, margins, validation_x, validation_y, settings):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    regressor = HistGradientBoostingRegressor(**settings).fit(train_x, margins)
    calibration = LogisticRegression().fit(regressor.predict(validation_x).reshape(-1, 1),
                                           validation_y)
    return regressor, calibration


def margin_tree(train_x, margins, validation_x, validation_y, test_x, settings):
    regressor, calibration = fit_margin_tree(train_x, margins, validation_x, validation_y, settings)
    return calibration.decision_function(regressor.predict(test_x).reshape(-1, 1))


TREE_SETTINGS = dict(max_iter=180, max_leaf_nodes=15, learning_rate=.05,
                     l2_regularization=2., early_stopping=False, random_state=17)


def trees(train, validation, test, adjusted):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    base = [np.stack([tabular(row) for row in rows]) for rows in (train, validation, test)]
    y = [np.asarray([row[3] for row in rows]) for rows in (train, validation, test)]
    margins = np.asarray([row[0]["score"][0] - row[0]["score"][1] for row in train])
    results = {}
    for extra in (False, True):
        x = base
        if extra:
            x = [np.column_stack((matrix, [adjusted[row[0]["id"]] for row in rows]))
                 for matrix, rows in zip(base, (train, validation, test))]
        classifier = HistGradientBoostingClassifier(**TREE_SETTINGS).fit(x[0], y[0])
        suffix = "_adjusted" if extra else ""
        results["trees_winner" + suffix] = score(y[2], classifier.decision_function(x[2]))
        results["trees_margin" + suffix] = score(
            y[2], margin_tree(x[0], margins, x[1], y[1], x[2], TREE_SETTINGS))
    return results


def rolling_margin_predictions(x, margins, truth, dates, train, validation, test):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    model = HistGradientBoostingRegressor(**TREE_SETTINGS).fit(x[train], margins[train])
    calibration = LogisticRegression().fit(
        model.predict(x[validation]).reshape(-1, 1), truth[validation])
    predictions = []
    completed = []
    refits = 0
    for _, day in groupby(np.flatnonzero(test), key=lambda index: dates[index]):
        indices = list(day)
        predictions.extend(calibration.decision_function(model.predict(x[indices]).reshape(-1, 1)))
        completed.extend(indices)
        # Refit once per day at most, after every pick for that date has been recorded.
        if len(completed) >= (refits + 1) * 32:
            eligible = train.copy()
            eligible[completed] = True
            model.fit(x[eligible], margins[eligible])
            refits = len(completed) // 32
    return predictions


def window_test(rows, rolling=False):
    import numpy as np

    x = np.stack([tabular(row) for row in rows])
    truth = np.asarray([row[3] for row in rows])
    margins = np.asarray([row[0]["score"][0] - row[0]["score"][1] for row in rows])
    seasons = np.asarray([int(row[0]["season"]) for row in rows])
    dates = [row[0]["date"] for row in rows]
    report = {}
    for year in range(int(seasons.min()) + 11, int(seasons.max()) + 1):
        validation = seasons == year - 1
        test = seasons == year
        report[str(year)] = {}
        windows = (("10", 10),) if rolling else (
            ("5", 5), ("10", 10), ("all", year - 1 - int(seasons.min())))
        for label, years in windows:
            train = (seasons >= year - 1 - years) & (seasons < year - 1)
            assert max(row[0]["date"] for row, use in zip(rows, train) if use) < min(
                row[0]["date"] for row, use in zip(rows, validation) if use)
            logits = margin_tree(x[train], margins[train], x[validation], truth[validation],
                                 x[test], TREE_SETTINGS)
            report[str(year)][label] = dict(score(truth[test], logits), training_seasons=years,
                                            training_games=int(train.sum()))
            if rolling:
                updated = rolling_margin_predictions(x, margins, truth, dates, train, validation, test)
                report[str(year)]["refit_32"] = score(truth[test], updated)
        print(year, report[str(year)], flush=True)
    output = "tree-refit-32" if rolling else "tree-training-windows"
    Path(f"data/{output}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("adaptive", "trees", "windows", "tree-refit"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    games = read_games("data/games-expanded.jsonl")
    rows = list(examples(games, EXPANDED_FEATURES))
    if args.phase in ("windows", "tree-refit"):
        window_test([row for row in rows if row[3] is not None], rolling=args.phase == "tree-refit")
        return
    adjusted = dict(adjusted_efficiency(games)) if args.phase == "trees" else {}
    report = {}
    for year in (2023, 2024):
        train, validation, test = split(rows, f"{year}-10-01", f"{year+1}-10-01")
        test = [row for row in test if row[0]["date"] < f"{year+2}-10-01"]
        if args.phase == "trees":
            results = trees(train, validation, test, adjusted)
        else:
            results = {}
            for interval, rate in ((0, 0.), (2, 3e-5), (16, 3e-5), (32, 3e-5),
                                   (2, 1e-5), (16, 1e-5), (32, 1e-5)):
                checkpoint = f"model-matchup-{year}-margin.pt"
                logits, steps = adaptive_predictions(checkpoint, train, test, interval, rate)
                name = f"update_{interval}_{rate:g}" if interval else "frozen"
                results[name] = dict(score([row[3] for row in test], logits), steps=steps)
        report[str(year + 1)] = results
        print(year + 1, json.dumps(results), flush=True)
        Path(f"data/training-comparison-{args.phase}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

