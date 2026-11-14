"""Generate a shareable HTML demo from a real held-out NBA prediction."""

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

from basketball import EXPANDED_FEATURES, examples, read_games, split
from compare_training import TREE_SETTINGS, fit_margin_tree, score, tabular


def feature_names():
    team = [
        "Recent win rate", "Rest days (capped)", "Recent game count", "Neutral site",
        "Season wins", "Season losses", "Recent scoring margin", "Back-to-back",
        "Rest history available", "Days until playoffs", "Playoff start known",
        "Regular season", "Play-in", "Playoffs", "Pace", "Offensive rating",
        "Defensive rating", "Efficiency sample size", "Usual starter minutes outside starting five",
        "Usual starter points outside starting five", "Rotation history size",
        "Prior starts together", "Shrinkage-adjusted margin when pairs started",
        "Games in last 4 days", "Games in last 7 days", "Games in last 14 days",
        "Consecutive road games",
    ]
    for metric in ("Effective field-goal percentage", "Turnover rate", "Offensive rebound rate",
                   "Free-throw attempts per field-goal attempt", "Team minutes", "Rim attempt share",
                   "Corner-three attempt share", "Three-point attempt share", "Opponent-adjusted margin"):
        for side in ("Team", "Opponent allowed"):
            team.extend((f"{side} {metric.lower()}", f"{side} {metric.lower()} sample size"))
    team.extend(f"Starter pair shared minutes, rank {i}" for i in range(1, 11))
    team.extend(("Team playoff series wins", "Opponent playoff series wins"))
    player = []
    for window in ("Last 10 appearances", "Season to date", "Career to date"):
        player.extend(f"{window} {metric}" for metric in
                      ("points per game", "rebounds per game", "assists per game", "minutes per game",
                       "appearance count"))
        if window == "Last 10 appearances":
            player.append("Starter status")
    player.extend(("Recent field-goal attempts", "Recent free-throw attempts"))
    player.extend(f"Prior starts with current starter, rank {i}" for i in range(1, 5))
    player.append("Shrinkage-adjusted team margin with current starter pairs")
    assert len(team) == 75 and len(player) == 23
    return team, player


def input_catalog(team, player):
    team_rules = [
        "Wins divided by games in the last 10 team games.",
        "Calendar days since the previous game, capped at 14 and divided by 14.",
        "Games available in the last-10 window, divided by 10.",
        "1 when the game is at a neutral venue, otherwise 0.",
        "Prior wins this season, divided by 82.",
        "Prior losses this season, divided by 82.",
        "Average points scored minus points allowed over the last 10 games, divided by 30.",
        "1 when the team played the previous day.",
        "1 when a previous game is available to calculate rest.",
        "Days to the published playoff start, clipped to 0–365 and divided by 365.",
        "1 when a playoff date known before the game is supplied.",
        *["1 if the scheduled game matches this phase." for _ in range(3)],
        "Last-10 average pace (possessions per 48 minutes), subtract 100, then divide by 20.",
        "Last-10 average points scored per 100 possessions, subtract 110, then divide by 20.",
        "Last-10 average points allowed per 100 possessions, subtract 110, then divide by 20.",
        "Available games in the efficiency window, divided by 10.",
        "Rank teammates by minutes over the last 10 games. Sum minutes for members of the usual top five outside today’s starting five, per game, divided by 180.",
        "Points from those same players over the same period, per game, divided by 100.",
        "Games with player rotation history in the last-10 window, divided by 10.",
        "For each starter pair, log(1 + prior starts together). Sum across pairs and divide by 70.",
        "For each pair, divide total team scoring margin in their prior starts by starts + 10. Sum across pairs and divide by 300.",
        *[f"Count games in the past {days} calendar days and divide by {days}." for days in (4, 7, 14)],
        "Consecutive prior away games, up to 20, divided by 20.",
    ]
    sources = [
        "(made field goals + 0.5 × made threes) ÷ field-goal attempts",
        "team turnover percentage from the team game log",
        "offensive rebounding percentage from the team game log",
        "free-throw attempts ÷ field-goal attempts",
        "total minutes played across all players",
        "share of charted field-goal attempts within four feet",
        "share of charted field-goal attempts classified as corner threes by court coordinates",
        "share of charted field-goal attempts worth three points",
        "actual scoring margin plus the prior opponent’s own recent scoring margin before that game",
    ]
    for source, (center, scale) in zip(sources, (
            (.52, .1), (.14, .1), (.25, .15), (.24, .2), (240, 25),
            (.3, .2), (.08, .1), (.38, .2), (0, 20))):
        for subject in ("the team itself", "the teams that previously faced today’s opponent"):
            team_rules.extend((
                f"Per game, measure {source}. Average up to 20 observations for {subject}; subtract {center} and divide by {scale}.",
                "Count nonmissing observations for the preceding average, then divide by 20.",
            ))
    team_rules.extend(
        [f"The #{rank} largest prior shared-minute total among the ten starter pairs. Sum overlapping stints, then take log(1 + minutes) ÷ 8."
         for rank in range(1, 11)] +
        ["Games already won in the current playoff series by this team, divided by 4.",
         "Games already won in the current playoff series by the opponent, divided by 4."])
    player_rules = []
    for window in ("last 10 appearances", "current season", "available career history"):
        player_rules.extend(f"Average {stat} over the {window}, divided by {scale}."
                            for stat, scale in zip(("points", "rebounds", "assists", "minutes"),
                                                  (30, 12, 10, 36)))
        player_rules.append("Appearances divided by 10." if window.startswith("last") else
                            f"log(1 + appearances) ÷ 8 over the {window}.")
        if window.startswith("last"):
            player_rules.append("1 for a starter, −1 for a listed nonstarter, 0 if unknown. This dataset supplies only starters.")
    player_rules.extend(("Last-10 average field-goal attempts divided by 20.",
                         "Last-10 average free-throw attempts divided by 10."))
    player_rules.extend(f"Rank {rank} by prior starts with the player among today’s other starters. Use log(1 + starts) ÷ 7."
                        for rank in range(1, 5))
    player_rules.append("Sum each current pair’s historical team margin divided by its prior starts + 10. Divide the sum by 120.")
    assert len(team_rules) == len(team) and len(player_rules) == len(player)
    tables = []
    for title, names, rules in (("75 team inputs per side", team, team_rules),
                               ("23 inputs per starter", player, player_rules)):
        rows = "".join(f"<tr><td>{i}</td><th scope='row'>{escape(name)}</th><td>{escape(rule)}</td></tr>"
                       for i, (name, rule) in enumerate(zip(names, rules), 1))
        tables.append(f"<h3>{title}</h3><table><thead><tr><th>#</th><th>Variable</th>"
                      f"<th>How it is calculated</th></tr></thead><tbody>{rows}</tbody></table>")
    return "".join(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", help="NBA game ID from the 2025–26 test set")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    rows = examples(read_games(project / "data/games-expanded.jsonl"), EXPANDED_FEATURES)
    train, calibration, test = split(rows, "2024-10-01", "2025-10-01")
    report = json.loads((project / "data/tree-training-windows.json").read_text())
    train = [row for row in train if int(row[0]["season"]) >= 2014]
    index = next((i for i, row in enumerate(test) if row[0]["id"] == args.game_id), None)
    if args.game_id and index is None:
        parser.error("Game ID is outside the test set.")
    index = index if index is not None else 0
    model, calibrator = fit_margin_tree(
        np.stack([tabular(row) for row in train]),
        np.asarray([row[0]["score"][0] - row[0]["score"][1] for row in train]),
        np.stack([tabular(row) for row in calibration]), [row[3] for row in calibration],
        TREE_SETTINGS,
    )
    test_x = np.stack([tabular(row) for row in test])
    predictions = model.predict(test_x)
    actual_margins = np.asarray([row[0]["score"][0] - row[0]["score"][1] for row in test])
    stages = [(i, float(np.abs(estimate - actual_margins).mean()))
              for i, estimate in enumerate(model.staged_predict(test_x), 1)
              if i in (1, 10, 30, 60, 90, 120, 150, 180)]
    team_names, player_names = feature_names()
    side_names = team_names + [f"Starters’ {stat} of {name.lower()}"
                               for stat in ("mean", "maximum", "minimum") for name in player_names]
    columns = [f"{side} · {name}" for side in ("Home", "Away", "Home minus away")
               for name in side_names]
    stage_rows = "".join(f"<tr><th scope='row'>{i}</th><td><meter min='0' max='{max(e for _, e in stages):.4f}' "
                         f"value='{mae:.4f}'></meter></td><td>{mae:.2f} pts</td></tr>" for i, mae in stages)
    criteria = []
    # Inspect the fitted tree nodes for illustration only. Prediction uses public APIs.
    for tree in model._predictors[:3]:
        root = tree[0].nodes[0]
        if not root["is_leaf"]:
            name = escape(columns[int(root["feature_idx"])])
            criteria.append(f"<li>{name} ≤ {root['num_threshold']:.3f}</li>")
    catalog = input_catalog(team_names, player_names)
    test_metrics = score([row[3] for row in test],
                         calibrator.decision_function(predictions.reshape(-1, 1)))
    totals = sum(year["10"]["games"] for year in report.values())
    correct = sum(year["10"]["correct"] for year in report.values())
    game, tokens, players, win, _ = test[index]
    home, away = (escape(game[side]["team"]) for side in ("home", "away"))
    probability = calibrator.predict_proba(predictions[index:index + 1].reshape(-1, 1))[0, 1]
    margin = predictions[index]
    pick = home if probability >= .5 else away
    result = "Correct pick" if (probability >= .5) == bool(win) else "Incorrect pick"
    metrics = []
    for side in (0, 1):
        team = next(values for name, token_side, values in tokens
                    if name.startswith("team/") and token_side == side)
        starter_points = sum(values[0] * 30 for _, token_side, values in players if token_side == side)
        metrics.append((round(team[4] * 82), round(team[5] * 82), team[1] * 14, starter_points))
    game_id = escape(game["id"])
    cards = "".join(
        f'<div><small>{int(year)}–{str(int(year)+1)[-2:]}</small>'
        f'<strong>{season["10"]["accuracy"]:.2%}</strong><meter min="0" max="1" '
        f'value="{season["10"]["accuracy"]}">{season["10"]["accuracy"]:.2%}</meter>'
        f'<small>{season["10"]["correct"]:,} / {season["10"]["games"]:,} picks</small></div>'
        for year, season in report.items()
    )
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Auto Regression Model NBA Test</title>
<style>
body {{font: 16px system-ui,sans-serif; background:#f4f5f8;color:#182332;margin:0}}
main {{max-width:1120px;margin:auto;padding:36px 20px}}
h1 {{font-size:clamp(2rem,5vw,3.4rem);margin:0}}
section {{background:white;border:1px solid #dde2ea;border-radius:16px;padding:24px;margin:20px 0}}
.grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:20px}}
strong {{display:block;font-size:1.65rem;color:#135743}}
small {{color:#46556a}} meter {{width:100%;accent-color:#19785b}}
table {{width:100%;border-collapse:collapse}} td,th {{padding:10px;text-align:left;border-bottom:1px solid #dde2ea}}
td,th {{vertical-align:top}} .catalog th[scope="row"] {{width:30%}}
details {{margin-top:20px}} summary {{cursor:pointer;font-weight:650}}
</style>
<main>
<h1>Auto Regression Model NBA Test</h1>
<p>The boosted trees learn point margins from <b>2014–15 through 2023–24</b>.
The <b>2024–25</b> season calibrates win probabilities. The model stays fixed for the <b>2025–26</b> test.</p>
<section class="catalog"><h2>Measured inputs and derived variables</h2>
<h3>Where the raw data comes from</h3>
<ul>
<li><b>Game records:</b> NBA game ID, date, season, phase, playoff round and series, home and away team IDs, neutral-site flag, and final scores.</li>
<li><b>Team game logs:</b> Pace, offensive rating, defensive rating, effective field-goal percentage, turnover rate, offensive rebounding rate, free-throw and field-goal attempts, and player-minutes summed by team.</li>
<li><b>Player game logs:</b> Player and team IDs, points, rebounds, assists, minutes, field-goal attempts, and free-throw attempts. Bench box scores contribute to earlier rotation and career history.</li>
<li><b>Start and rotation records:</b> Starter IDs and on/off court times by player. Older starting fives come from opening stints where those stints agree with the box score.</li>
<li><b>Play-by-play shots:</b> Shot value, distance and court coordinates, used to build prior shot profiles.</li>
</ul>
<p>The importer joins the <a href="https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md">NBA data archive</a> by game, team and player IDs. It writes <code>games-expanded.jsonl</code>. The feature builder processes that file in date order. The final score supplies the training target and can update stats for later dates. A game's own box score and play-by-play stay out of its inputs.</p>
<h3>How the variables reach the trees</h3>
<p>For each side, keep the 75 team values and calculate the <b>mean, maximum, and minimum</b> across the starters for each of the 23 player values. That gives 144 numbers per side. Place the home values, away values, and home-minus-away differences side by side for <b>432 columns</b>. For example, the model can split on the away team's rest, the highest starter scoring average, or the difference between the teams’ typical scoring margins.</p>
<p>Below, each row describes the value before the home/away comparison or starter summary. Missing history generally contributes zero, with sample counts or known flags where available. The optional playoff start date has no values in this dataset. Recorded starters are treated as known before tipoff.</p>
{catalog}
</section>
<section><h2>How it works</h2>
<ol>
<li><b>Build the pregame snapshot.</b> Walk through games by date and calculate each day's inputs before adding its results to history. The replay assumes recorded starters were known at tipoff.</li>
<li><b>Turn two lineups into one row.</b> Each side contributes 75 team inputs and 69 starter summaries. The summaries are the mean, maximum, and minimum of each of 23 player inputs. Add 144 home-minus-away differences for 432 inputs.</li>
<li><b>Learn point margin.</b> Train on {len(train):,} games. The target is final home points minus final away points. Each of 180 trees uses earlier errors to learn where to adjust the predicted margin. The learning rate is 0.05, and each tree has up to 15 leaves.</li>
<li><b>Convert margin to a win chance.</b> Logistic calibration uses {len(calibration):,} games from 2024–25. It learns a probability curve from predicted margins and observed winners. Pick home at 50% or higher.</li>
<li><b>Replay the test season.</b> The fitted trees and calibration stay fixed. Completed games refresh the inputs for later dates.</li>
</ol>
<h3>Criteria learned from the data</h3>
<p>These are the opening splits of the first three fitted trees. Thresholds use the model’s scaled inputs. A path through a tree combines several criteria, and each leaf contributes a margin adjustment.</p>
<ul>{''.join(criteria)}</ul>
</section>
<section><h2>Follow one real prediction</h2>
<p>{escape(game["date"])} · NBA game {game_id} · {away} away at {home} home</p>
<table><thead><tr><th>Sample pregame inputs</th><th>Home</th><th>Away</th></tr></thead><tbody>
<tr><th scope="row">Season record</th><td>{metrics[0][0]}–{metrics[0][1]}</td><td>{metrics[1][0]}–{metrics[1][1]}</td></tr>
<tr><th scope="row">Days since previous game (capped at 14)</th><td>{metrics[0][2]:.0f}</td><td>{metrics[1][2]:.0f}</td></tr>
<tr><th scope="row">Sum of starters’ last-10 points averages</th><td>{metrics[0][3]:.1f}</td><td>{metrics[1][3]:.1f}</td></tr>
</tbody></table>
<h3>What the model predicted</h3><div class="grid">
<div><small>Predicted home margin</small><strong>{margin:+.1f} pts</strong></div>
<div><small>Home win probability</small><strong>{probability:.1%}</strong></div>
<div><small>Picked team ID</small><strong>{pick}</strong></div></div>
<details><summary>Reveal the final score</summary>
<p>Home {game["score"][0]} · Away {game["score"][1]} · {result}</p></details>
</section>
<section><h2>Winner accuracy across four historical tests</h2>
<p><b>{correct:,} of {totals:,} correct ({correct / totals:.2%})</b> · Each test used its own earlier ten-season training window and one calibration season.</p>
<div class="grid">{cards}</div>
<p>This demo refit reproduced {test_metrics["correct"]:,} of {test_metrics["games"]:,}
correct picks ({test_metrics["accuracy"]:.2%}) on 2025–26.</p></section>
<section><h2>Learning as trees are added</h2>
<p>Average absolute point-margin error on the same 2025–26 games as more trees contribute their corrections. Lower is better.</p>
<table><thead><tr><th>Trees used</th><th>Error magnitude</th><th>Average error</th></tr></thead><tbody>{stage_rows}</tbody></table>
<p>The four season results above come from separate fits with rolling training windows. The table here follows one fit as it adds trees.</p>
</section>
<section><h2>Where to look in the code</h2>
<ol><li><code>basketball.py</code> builds features from earlier games.</li>
<li><code>compare_training.py</code> builds the matchup row, trains the trees and calibrates probabilities.</li>
<li><code>show_model.py</code> runs the model and creates this report.</li></ol></section>
</main></html>
"""
    output = project / "data/model-demo.html"
    output.write_text(html, encoding="utf-8")
    print(f"""
NBA WINNER MODEL
================

1. BUILD THE PREGAME INPUTS
   Each JSONL record describes one game. The feature builder walks through games
   by date and calculates rolling stats before adding that day's results to
   history. This keeps the target game's score out of its own inputs.
   The historical backtest assumes the recorded starters were known at tipoff.

   Each matchup becomes {tabular(test[index]).size} numbers. For each side we
   include 75 team features and the mean, maximum, and minimum of 23 starter
   features. Then we add the home-minus-away differences. Examples include
   recent scoring, rest, shooting efficiency, and shared floor time.

2. TRAIN THE BOOSTED TREES
   2014-15 through 2023-24 | {len(train):,} games
   The training target is final home score minus final away score.
   The model starts with an average margin and fits {model.n_iter_} trees in
   sequence. Each tree learns a small correction from the remaining errors.
   For instance, a tree can learn different corrections for teams with and
   without a rest advantage. The model learns splits and thresholds from data.
   Each tree has up to {TREE_SETTINGS['max_leaf_nodes']} leaves, and its
   correction is scaled by a learning rate of {TREE_SETTINGS['learning_rate']}.

3. CALIBRATE WIN PROBABILITIES
   2024-25 | {len(calibration):,} games
   A logistic regression learns how predicted margins map to actual wins.
   Home win probability = sigmoid({calibrator.intercept_[0]:+.4f}
                                 {calibrator.coef_[0, 0]:+.4f} * predicted margin)
   Pick home at 50% or higher. Otherwise pick away.

4. FOLLOW ONE HELD-OUT GAME
   {game['date']} | Game {game['id']} | Away {away} at home {home}
                                   Home       Away
   Pregame wins / losses           {metrics[0][0]:>2}/{metrics[0][1]:<2}      {metrics[1][0]:>2}/{metrics[1][1]:<2}
   Days since previous game*       {metrics[0][2]:>5.0f}      {metrics[1][2]:>5.0f}
   Sum of starters' last-10 PPG     {metrics[0][3]:>5.1f}      {metrics[1][3]:>5.1f}
   *Capped at 14 days.

   Predicted home margin           {margin:+.1f} points
   Home / away win chance          {probability:.1%} / {1-probability:.1%}
   Picked team ID                  {pick}
   Final score                     Home {game['score'][0]}, away {game['score'][1]}
   Outcome                         {result}

5. SCORE EVERY ELIGIBLE TEST GAME
   2025-26 | {test_metrics['correct']:,} / {test_metrics['games']:,} correct ({test_metrics['accuracy']:.2%})
   Win-probability log loss: {test_metrics['log_loss']:.4f} (lower is better).
   The trees and calibrator stay fixed. Earlier completed games can still
   update the input stats for later matchups.

   Saved four-season comparison (a separate fit for each test season)
   Season       Correct / Games      Accuracy""")
    for year, season in sorted(report.items()):
        row = season["10"]
        print(f"   {year}-{str(int(year)+1)[-2:]}       {row['correct']:,} / {row['games']:,}         {row['accuracy']:.2%}")
    print(f"   Overall       {correct:,} / {totals:,}       {correct / totals:.2%}")
    print(f"""
CODE TOUR
   basketball.py        Build chronological pregame features.
   context_features.py  Track opponent tendencies and prior shared floor time.
   compare_training.py  Flatten features, fit trees, and calibrate probabilities.
   show_model.py        Run the demonstration and report the results.

Browser report
   {output}
""")


if __name__ == "__main__":
    main()

