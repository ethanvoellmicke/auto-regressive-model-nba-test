"""Continue a saved transformer on newly completed games with historical replay."""

import argparse
from datetime import date
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from basketball import examples, read_games
from transformer import batch, evaluate, load_checkpoint, vocabulary


def update(games_path, source_path, output_path, after):
    if source_path.resolve() == output_path.resolve() or output_path.exists():
        raise ValueError("Save to a new checkpoint path and preserve the source.")
    torch.manual_seed(17)
    randomizer = random.Random(17)
    model, saved = load_checkpoint(source_path)
    cutoff = saved.get("updated_through", after)
    if cutoff is None:
        raise ValueError("The first incremental update requires --after.")
    date.fromisoformat(cutoff)
    rows = list(examples(read_games(games_path), model.features.in_features))
    fresh = [row for row in rows if row[3] is not None and row[0]["date"] > cutoff]
    if not fresh:
        return {"new_games": 0, "updated_through": cutoff, "saved": False}
    if fresh[0][0]["date"] <= saved["selected_through"]:
        raise ValueError("New training games must follow prior checkpoint selection.")
    replay = [row for row in rows if row[3] is not None
              and row[0]["date"] <= saved["trained_through"]]
    if not replay:
        raise ValueError("Historical replay games are required.")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    vocab = dict(saved["vocab"])
    # New identities start with the existing unknown vector. All learned vectors
    # and the original ID assignments remain unchanged before optimization.
    additions = sorted(set(vocabulary(fresh)) - set(vocab))
    if additions:
        weight = model.identity.weight.detach()
        extension = weight[0:1].repeat(len(additions), 1)
        model.identity = nn.Embedding.from_pretrained(
            torch.cat((weight, extension)), freeze=False
        )
        for name in additions:
            vocab[name] = len(vocab)
    baseline = saved.get("baseline", .5)
    model.eval()
    before_metrics = evaluate(model, fresh, vocab, torch.device("cpu"), 32,
                              baseline, winner_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    steps = 0
    for start in range(0, len(fresh), 16):
        chunk = fresh[start:start + 16]
        historical = randomizer.sample(replay, min(len(chunk), len(replay)))
        inputs = batch(chunk + historical, vocab, torch.device("cpu"), winner_only=True)
        winner, margin = model(*inputs[:6], winner_only=True, return_margin=True)
        actual = winner.new_tensor([
            (row[0]["score"][0] - row[0]["score"][1]) / 15
            for row in chunk + historical
        ])
        loss = F.binary_cross_entropy_with_logits(winner, inputs[-2])
        loss = loss + .25 * F.smooth_l1_loss(margin, actual)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        steps += 1
    model.eval()
    after_metrics = evaluate(model, fresh, vocab, torch.device("cpu"), 32,
                             baseline, winner_only=True)
    squared_change = 0.
    squared_original = 0.
    for name, original in before.items():
        current = model.state_dict()[name]
        if name == "identity.weight":
            current = current[:original.shape[0]]
        squared_change += (current - original).square().sum().item()
        squared_original += original.square().sum().item()
    report = {
        "source_checkpoint": source_path.name,
        "new_games": len(fresh),
        "new_from": fresh[0][0]["date"],
        "updated_through": fresh[-1][0]["date"],
        "learning_rate": 1e-5,
        "new_games_per_batch": 16,
        "historical_replay_ratio": "1:1",
        "optimizer_steps": steps,
        "new_identities": len(additions),
        "relative_weight_change": (squared_change / squared_original) ** .5,
        "before_update_unseen_games": before_metrics,
        "after_update_training_fit_only": after_metrics,
        "saved": True,
    }
    saved.update(
        weights=model.state_dict(), vocab=vocab,
        trained_through=fresh[-1][0]["date"],
        selected_through=fresh[-1][0]["date"],
        updated_through=fresh[-1][0]["date"],
        original_checkpoint=saved.get("original_checkpoint", source_path.name),
        source_checkpoint=source_path.name, incremental_update=report,
        margin_weight=.25,
    )
    torch.save(saved, output_path)
    loaded, _ = load_checkpoint(output_path)
    inputs = batch(fresh[-1:], vocab, torch.device("cpu"), winner_only=True)
    with torch.no_grad():
        expected = model(*inputs[:6], winner_only=True)
        actual = loaded(*inputs[:6], winner_only=True)
    if not torch.equal(expected, actual) or not torch.isfinite(actual).all():
        raise ValueError("Saved checkpoint inference did not reproduce.")
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--after", help="Initial last-processed date, YYYY-MM-DD.")
    args = parser.parse_args()
    print(json.dumps(update(args.games, args.source, args.output, args.after), indent=2))


if __name__ == "__main__":
    main()

