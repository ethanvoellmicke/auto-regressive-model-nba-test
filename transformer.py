"""Factor the forecast into a winner and stat distributions given the winner."""

import copy
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from basketball import CAPS, EXPANDED_FEATURES, FEATURES, RECENT_GAME_FEATURES, STATS


SPECIAL = ("unknown", "winner", "home_won", "away_won")


def vocabulary(rows):
    names = {name for _, teams, players, _, _ in rows for name, _, _ in teams}
    names.update(f"player/{player}" for _, _, players, _, _ in rows for player, _, _ in players)
    return {name: index for index, name in enumerate((*SPECIAL, *sorted(names)))}


def batch(rows, vocab, device, assumed_win=None, winner_only=False):
    feature_count = len(rows[0][1][0][2])
    sequences, winner_positions, player_positions, wins, labels = [], [], [], [], []
    for index, (_, teams, players, win, targets) in enumerate(rows):
        roster = [(f"player/{player}", side, stats) for player, side, stats in players]
        outcome = win if assumed_win is None else assumed_win
        prefix = teams + roster
        winner_positions.append(len(prefix))
        prefix += [("winner", 2, [0] * feature_count)]
        if winner_only:
            sequences.append(prefix)
            wins.append(win if win is not None else 0)
            continue
        prefix += [("home_won" if outcome else "away_won", 2, [0] * feature_count)]
        player_positions.extend((index, len(prefix) + i) for i in range(len(roster)))
        sequences.append(prefix + roster)
        wins.append(win if win is not None else 0)
        labels.extend(target if target is not None else [-100] * len(STATS) for target in targets)
    width = max(map(len, sequences))
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    sides = torch.full_like(ids, 2)
    stats = torch.zeros(len(rows), width, feature_count)
    padding = torch.ones_like(ids, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        size = len(sequence)
        names, team_sides, features = zip(*sequence)
        ids[index, :size] = torch.tensor([vocab.get(name, 0) for name in names])
        sides[index, :size] = torch.tensor(team_sides)
        stats[index, :size] = torch.tensor(features)
        padding[index, :size] = False
    return tuple(t.to(device) for t in (
        ids, sides, stats, padding, torch.tensor(winner_positions),
        torch.tensor(player_positions, dtype=torch.long).reshape(-1, 2),
        torch.tensor(wins, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long).reshape(-1, len(STATS)),
    ))


class Forecaster(nn.Module):
    def __init__(self, vocab_size, width=64, feature_count=FEATURES, token_layers=0,
                 attention_layers=2, bidirectional=False, team_comparison=False, margin_loss=False):
        super().__init__()
        self.identity = nn.Embedding(vocab_size, width)
        self.side = nn.Embedding(3, width)
        self.features = nn.Linear(feature_count, width)
        self.token_layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            for _ in range(token_layers)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, batch_first=True,
                                       norm_first=True, activation="gelu"),
            attention_layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.winner = nn.Linear(width, 1)
        self.bidirectional = bidirectional
        self.comparison = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1)) if team_comparison else None
        self.margin = nn.Linear(width, 1) if margin_loss else None
        self.total = nn.Linear(width, 2)
        self.minutes = nn.Linear(width, 1)
        self.stat_heads = nn.ModuleList(nn.Linear(width, cap + 1) for cap in CAPS)
        for head in [self.total, self.minutes, *self.stat_heads]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, ids, sides, stats, padding, winner_positions, player_positions, winner_only=False,
                return_margin=False):
        if self.training:
            ids = ids.masked_fill((ids >= len(SPECIAL)) & (torch.rand_like(stats[..., 0]) < .05), 0)
        hidden = self.identity(ids) + self.side(sides) + self.features(stats)
        for layer in self.token_layers:
            hidden = hidden + layer(hidden)
        causal = None
        if not (winner_only and self.bidirectional):
            causal = torch.ones(ids.size(1), ids.size(1), device=ids.device, dtype=torch.bool).triu(1)
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=padding)
        winners = hidden[torch.arange(ids.size(0), device=ids.device), winner_positions]
        winner = self.winner(winners).squeeze(-1)
        if winner_only:
            if self.comparison is not None:
                pooled = []
                for side in (0, 1):
                    mask = ((sides == side) & ~padding).unsqueeze(-1)
                    pooled.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
                home, away = pooled
                winner = winner + self.comparison(torch.cat((home, away, home-away, home*away), -1)).squeeze(-1)
            if return_margin:
                return winner, self.margin(winners).squeeze(-1)
            return winner
        players = hidden[player_positions[:, 0], player_positions[:, 1]]
        player_features = stats[player_positions[:, 0], player_positions[:, 1]]
        recent = torch.where(player_features[:, 4:5] > 0, player_features[:, :4],
                             player_features[:, 11:15])
        defaults = recent.new_tensor([.4, .4, .3, .8])
        recent = torch.where((player_features[:, 4:5] + player_features[:, 15:16]) > 0,
                             recent, defaults)
        total = self.total(winners)
        minutes = (recent[:, 3] * 36 + 5 * self.minutes(players).squeeze(-1)).clamp(1, 65)
        distributions = []
        for i, (head, cap, scale, spread) in enumerate(zip(self.stat_heads, CAPS, (30, 12, 10), (8, 3, 3))):
            center = recent[:, i] * scale * minutes / (recent[:, 3] * 36).clamp_min(1)
            values = torch.arange(cap + 1, device=ids.device)
            prior = (-.5 * ((values - center[:, None]) / spread).square()).clamp_min(-15)
            distributions.append(head(players) + prior)
        return (winner, distributions, 220 + 20 * total[:, 0],
                20 * total[:, 1].clamp(-1, 1).exp(), minutes)


def forecast_batch(model, rows, vocab, device):
    home = batch(rows, vocab, device, assumed_win=1)
    away = batch(rows, vocab, device, assumed_win=0)
    winner, home_stats, total, deviation, _ = model(*home[:6])
    _, away_stats, _, _, _ = model(*away[:6])
    win_probability = winner.sigmoid()
    weights = win_probability[home[5][:, 0], None]
    distributions = [weights * h.softmax(-1) + (1 - weights) * a.softmax(-1)
                     for h, a in zip(home_stats, away_stats)]
    return winner, distributions, home[-2], home[-1], total, deviation


def total_over_probability(mean, deviation, line):
    p_over = .5 * math.erfc((math.floor(line) + .5 - mean) / deviation / math.sqrt(2))
    p_under = .5 * math.erfc((mean - math.ceil(line) + .5) / deviation / math.sqrt(2))
    return p_over / (p_over + p_under)


def implied_probability(price):
    return -price / (100 - price) if price < 0 else 100 / (100 + price)


def binary_log_loss(over, probability):
    return -math.log(max(probability if over else 1 - probability, 1e-9))


@torch.no_grad()
def evaluate(model, rows, vocab, device, batch_size, baseline, winner_only=False):
    model.eval()
    log_loss, correct = 0.0, 0
    stat_errors = [0.0] * len(STATS)
    rolling_errors = [0.0] * len(STATS)
    appearances = 0
    total_errors, market_errors, matched_errors, market_loss = 0., 0., 0., 0.
    matched, graded, correct_totals = 0, 0, 0
    sources = {}
    priced = graded_priced = 0
    priced_model_loss = priced_market_loss = 0.
    priced_net = 0.
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        if winner_only:
            inputs = batch(subset, vocab, device, winner_only=True)
            logits = model(*inputs[:6], winner_only=True)
            wins = inputs[-2]
            log_loss += F.binary_cross_entropy_with_logits(logits, wins, reduction="sum").item()
            correct += ((logits >= 0) == wins.bool()).sum().item()
            continue
        logits, distributions, wins, labels, totals, deviation = forecast_batch(model, subset, vocab, device)
        for row, mean, stddev in zip(subset, totals.tolist(), deviation.tolist()):
            game = row[0]
            final = sum(game["score"])
            total_errors += abs(mean - final)
            quote = game.get("draftkings_total")
            if quote:
                priced += 1
                book_line, over_price, under_price = quote
                if final != book_line:
                    graded_priced += 1
                    over_implied = implied_probability(over_price)
                    under_implied = implied_probability(under_price)
                    actual_over = final > book_line
                    priced_market_loss += binary_log_loss(
                        actual_over, over_implied / (over_implied + under_implied))
                    priced_model_loss += binary_log_loss(
                        actual_over, total_over_probability(mean, stddev, book_line))
                    pick_over = mean >= book_line
                    price = over_price if pick_over else under_price
                    payout = price / 100 if price > 0 else 100 / -price
                    priced_net += payout if pick_over == actual_over else -1
            line = game.get("close_total")
            if line is None:
                continue
            matched += 1
            sources[game["total_source"]] = sources.get(game["total_source"], 0) + 1
            market_errors += abs(line - final)
            matched_errors += abs(mean - final)
            if line != final:
                graded += 1
                over = final > line
                correct_totals += (mean >= line) == over
                market_loss += binary_log_loss(over, total_over_probability(mean, stddev, line))
        log_loss += F.binary_cross_entropy_with_logits(logits, wins, reduction="sum").item()
        correct += ((logits >= 0) == wins.bool()).sum().item()
        available = labels[:, 0] >= 0
        appearances += available.sum().item()
        rolling = torch.tensor([features[:3] for _, _, players, _, _ in subset
                                for _, _, features in players], device=device).reshape(-1, 3)
        for i, distribution in enumerate(distributions):
            mean = distribution @ torch.arange(CAPS[i] + 1, device=device, dtype=distribution.dtype)
            stat_errors[i] += (mean[available] - labels[available, i]).abs().sum().item()
            rolling_mean = rolling[:, i] * (30, 12, 10)[i]
            rolling_errors[i] += (rolling_mean[available] - labels[available, i]).abs().sum().item()
    home_wins = sum(row[3] for row in rows)
    winner_metrics = {
        "games": len(rows), "winner_log_loss": log_loss / len(rows),
        "winner_accuracy": correct / len(rows),
        "historical_home_rate_log_loss": (
            -home_wins * math.log(baseline) - (len(rows) - home_wins) * math.log1p(-baseline)
        ) / len(rows),
    }
    if winner_only:
        return winner_metrics
    return {
        "games": len(rows), "player_appearances": appearances,
        "game_total_mae": total_errors / len(rows),
        "draftkings_totals": {
            "quoted": priced, "pushes": priced - graded_priced,
            "model_log_loss": priced_model_loss / graded_priced if graded_priced else None,
            "no_vig_log_loss": priced_market_loss / graded_priced if graded_priced else None,
            "flat_unit_return": priced_net / priced if priced else None,
        },
        "closing_totals": {
            "matched": matched, "sources": sources, "pushes": matched - graded,
            "model_mae": matched_errors / matched if matched else None,
            "closing_line_mae": market_errors / matched if matched else None,
            "side_accuracy": correct_totals / graded if graded else None,
            "side_log_loss": market_loss / graded if graded else None,
        },
        "winner_log_loss": log_loss / len(rows), "winner_accuracy": correct / len(rows),
        "historical_home_rate_log_loss": (
            -home_wins * math.log(baseline) - (len(rows) - home_wins) * math.log1p(-baseline)
        ) / len(rows),
        "player_mae": dict(zip(STATS, [error / appearances if appearances else None for error in stat_errors])),
        "rolling_average_mae": dict(zip(STATS, [error / appearances if appearances else None for error in rolling_errors])),
    }


def fit(splits, args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    train, validation, test = splits
    if not train:
        raise ValueError("Training requires eligible completed games.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = vocabulary(train)
    margin_weight = args.margin_weight
    if margin_weight is None:
        margin_weight = .25 if args.margin_loss else 0.
    if not math.isfinite(margin_weight) or margin_weight < 0:
        raise ValueError("Margin weight must be finite and nonnegative.")
    use_margin = margin_weight > 0
    options = dict(bidirectional=args.bidirectional, team_comparison=args.team_comparison,
                   margin_loss=use_margin)
    model = Forecaster(len(vocab), args.width, len(train[0][1][0][2]),
                       args.token_layers, args.attention_layers, **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    baseline = (sum(row[3] for row in train) + 1) / (len(train) + 2)
    best_score = math.inf
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for start in range(0, len(order), args.batch_size):
            rows = [train[i] for i in order[start:start + args.batch_size]]
            inputs = batch(rows, vocab, device, winner_only=True)
            output = model(*inputs[:6], winner_only=True, return_margin=use_margin)
            winner = output[0] if use_margin else output
            loss = F.binary_cross_entropy_with_logits(winner, inputs[-2])
            if use_margin:
                margins = winner.new_tensor([(row[0]["score"][0] - row[0]["score"][1]) / 15 for row in rows])
                loss = loss + margin_weight * F.smooth_l1_loss(output[1], margins)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if not validation:
            print(f"Epoch {epoch + 1}/{args.epochs} complete", file=sys.stderr, flush=True)
            continue
        metrics = evaluate(model, validation, vocab, device, args.batch_size, baseline, winner_only=True)
        score = metrics["winner_log_loss"]
        print(f"Epoch {epoch + 1}/{args.epochs} validation winner log loss {score:.4f}",
              file=sys.stderr, flush=True)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_validation = metrics
    if validation:
        model.load_state_dict(best_state)
        report = {"validation": best_validation,
                  "test": evaluate(model, test, vocab, device, args.batch_size, baseline, winner_only=True)}
    else:
        report = {"training_games": len(train), "epochs": args.epochs,
                  "trained_from": train[0][0]["date"], "trained_through": train[-1][0]["date"]}
    torch.save({"weights": model.state_dict(), "vocab": vocab, "trained_through": train[-1][0]["date"],
                "selected_through": (validation or train)[-1][0]["date"], "baseline": baseline,
                "winner_only": True, "token_layers": args.token_layers,
                "attention_layers": args.attention_layers, "options": options,
                "margin_weight": margin_weight, "seed": args.seed}, args.checkpoint)
    return report


@torch.no_grad()
def load_checkpoint(path):
    device = torch.device("cpu")
    saved = torch.load(path, map_location=device, weights_only=True)
    feature_count = saved["weights"]["features.weight"].shape[1]
    if feature_count not in (FEATURES, EXPANDED_FEATURES, RECENT_GAME_FEATURES):
        raise ValueError("This checkpoint uses an older feature set. Retrain it with basketball.py train.")
    vocab = saved["vocab"]
    width = saved["weights"]["identity.weight"].shape[1]
    model = Forecaster(len(vocab), width, feature_count, saved.get("token_layers", 0),
                       saved.get("attention_layers", 2), **saved.get("options", {})).eval()
    model.load_state_dict(saved["weights"])
    return model, saved


def evaluate_checkpoint(rows, args, checkpoint):
    model, saved = checkpoint
    start = args.test_start.isoformat()
    end = args.test_end.isoformat() if args.test_end else None
    if end and end <= start:
        raise ValueError("Test end must follow test start.")
    if start <= saved["selected_through"]:
        raise ValueError("Test must start after checkpoint validation.")
    training = []
    test = []
    for row in rows:
        if end and row[0]["date"] >= end:
            break
        if row[3] is None:
            continue
        if row[0]["date"] <= saved["trained_through"]:
            training.append(row[3])
        if row[0]["date"] >= start:
            test.append(row)
    if not training or not test:
        raise ValueError("Evaluation input must include the original training history and test games.")
    baseline = saved.get("baseline", (sum(training) + 1) / (len(training) + 2))
    if args.ensemble:
        return evaluate_ensemble(load_ensemble(checkpoint, args.ensemble), test)
    return evaluate(model, test, saved["vocab"], torch.device("cpu"), 32, baseline,
                    winner_only=saved.get("winner_only", False))


def load_ensemble(checkpoint, paths):
    model, saved = checkpoint
    members = [checkpoint, *(load_checkpoint(path) for path in paths)]
    if any(not item.get("winner_only") or item["selected_through"] != saved["selected_through"]
           or net.features.in_features != model.features.in_features for net, item in members):
        raise ValueError("Ensemble members must share the feature set and validation cutoff.")
    return members


@torch.no_grad()
def evaluate_ensemble(members, rows):
    correct = 0
    loss = 0.
    for start in range(0, len(rows), 64):
        subset = rows[start:start + 64]
        probabilities = []
        for model, saved in members:
            inputs = batch(subset, saved["vocab"], torch.device("cpu"), winner_only=True)
            probabilities.append(model(*inputs[:6], winner_only=True).sigmoid())
        prediction = torch.stack(probabilities).mean(0)
        wins = inputs[-2]
        correct += ((prediction >= .5) == wins.bool()).sum().item()
        loss += F.binary_cross_entropy(prediction, wins, reduction="sum").item()
    return {"games": len(rows), "members": len(members), "correct": correct,
            "winner_accuracy": correct / len(rows), "winner_log_loss": loss / len(rows)}


@torch.no_grad()
def forecast(row, args, checkpoint):
    model, saved = checkpoint
    if row[0]["date"] <= saved["selected_through"]:
        raise ValueError("The forecast date must be after the checkpoint's validation data.")
    vocab = saved["vocab"]
    device = torch.device("cpu")
    if saved.get("winner_only"):
        if args.prop:
            raise ValueError("This checkpoint predicts game winners only.")
        members = load_ensemble(checkpoint, getattr(args, "ensemble", ()))
        probabilities = []
        for member, metadata in members:
            inputs = batch([row], metadata["vocab"], device, winner_only=True)
            probabilities.append(member(*inputs[:6], winner_only=True).sigmoid().item())
        probability = sum(probabilities) / len(probabilities)
        return {"game_id": row[0]["id"], "home_win_probability": probability,
                "selected_through": saved["selected_through"]}
    logits, distributions, _, _, total, deviation = forecast_batch(model, [row], vocab, device)
    players = [player_id for player_id, _, _ in row[2]]
    means = [dist @ torch.arange(cap + 1, dtype=dist.dtype)
             for dist, cap in zip(distributions, CAPS)]
    probabilities = []
    for player, stat, raw_line in args.prop:
        line = float(raw_line)
        if player not in players or stat not in STATS or not math.isfinite(line):
            raise ValueError(f"Invalid prop for player {player}, stat {stat}, line {raw_line}.")
        dist = distributions[STATS.index(stat)][players.index(player)]
        values = torch.arange(len(dist), dtype=torch.float64)
        probabilities.append({"player": player, "stat": stat, "line": line,
                              "over": dist[values > line].sum().item(),
                              "under": dist[values < line].sum().item(),
                              "push": dist[values == line].sum().item()})
    return {
        "game_id": row[0]["id"], "home_win_probability": logits.sigmoid().item(),
        "expected_game_total": total.item(), "game_total_stddev": deviation.item(),
        "selected_through": saved["selected_through"],
        "player_means_given_appearance": {
            player: {stat: values[i].item() for stat, values in zip(STATS, means)}
            for i, player in enumerate(players)
        },
        "prop_probabilities_given_appearance": probabilities,
    }

