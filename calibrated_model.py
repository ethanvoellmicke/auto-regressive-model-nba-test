"""The fixed, historically tested point-margin trees and probability calibrator."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from basketball import EXPANDED_FEATURES, examples, read_games, split
from compare_training import TREE_SETTINGS, fit_margin_tree, score, tabular


def fit_checkpoint(games_path, checkpoint_path):
    rows = list(examples(read_games(games_path), EXPANDED_FEATURES))
    train, calibration, test = split(rows, "2024-10-01", "2025-10-01")
    train = [row for row in train if int(row[0]["season"]) >= 2014]
    test = [row for row in test if row[0]["date"] < "2026-10-01"]
    model, calibrator = fit_margin_tree(
        np.stack([tabular(row) for row in train]),
        np.asarray([row[0]["score"][0] - row[0]["score"][1] for row in train]),
        np.stack([tabular(row) for row in calibration]),
        [row[3] for row in calibration],
        TREE_SETTINGS,
    )
    metrics = score(
        [row[3] for row in test],
        calibrator.decision_function(
            model.predict(np.stack([tabular(row) for row in test])).reshape(-1, 1)
        ),
    )
    if metrics["correct"] != 907 or metrics["games"] != 1316:
        raise ValueError("The original 907/1316 benchmark was not reproduced.")
    saved = {
        "model": model,
        "calibrator": calibrator,
        "model_type": "calibrated_point_margin_trees",
        "feature_count": EXPANDED_FEATURES,
        "training_games": len(train),
        "calibration_games": len(calibration),
        "trained_from": train[0][0]["date"],
        "trained_through": train[-1][0]["date"],
        "selected_through": calibration[-1][0]["date"],
        "settings": TREE_SETTINGS,
        "benchmark": metrics,
    }
    joblib.dump(saved, checkpoint_path)
    return saved


def load_checkpoint(path):
    saved = joblib.load(path)
    if saved.get("model_type") != "calibrated_point_margin_trees":
        raise ValueError("Expected the calibrated point-margin tree checkpoint.")
    return saved


def forecast_rows(rows, saved):
    if not rows:
        return []
    if any(row[0]["date"] <= saved["selected_through"] for row in rows):
        raise ValueError("Forecasts must follow the calibration period.")
    margins = saved["model"].predict(np.stack([tabular(row) for row in rows]))
    probabilities = saved["calibrator"].predict_proba(margins.reshape(-1, 1))[:, 1]
    return [
        {
            "game_id": row[0]["id"],
            "predicted_home_margin": float(margin),
            "home_win_probability": float(probability),
            "model_type": saved["model_type"],
            "trained_through": saved["trained_through"],
            "calibrated_through": saved["selected_through"],
        }
        for row, margin, probability in zip(rows, margins, probabilities)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("games", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--game-id")
    args = parser.parse_args()
    if args.command == "fit":
        saved = fit_checkpoint(args.games, args.checkpoint)
        print(json.dumps({key: value for key, value in saved.items()
                          if key not in ("model", "calibrator")}, indent=2))
        return
    if not args.game_id:
        parser.error("predict requires --game-id")
    saved = load_checkpoint(args.checkpoint)
    rows = [
        row for row in examples(read_games(args.games), saved["feature_count"])
        if row[0]["id"] == args.game_id
    ]
    if not rows:
        parser.error("Game ID was not found.")
    print(json.dumps(forecast_rows(rows, saved)[0], indent=2))


if __name__ == "__main__":
    main()

