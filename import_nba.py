"""Convert billmill NBA season parquet exports into the model's JSONL format."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import duckdb

# ESPN franchise IDs 1–30, in order, mapped to NBA franchise IDs.
ESPN_TEAMS = (2737, 2738, 2740, 2741, 2739, 2742, 2743, 2765, 2744, 2745,
              2754, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2755,
              2756, 2757, 2758, 2759, 2760, 2762, 2764, 2761, 2763, 2766)

# Standard UTC offsets for US/Canadian NBA venue cities.
VENUE_ZONES = {
    city: offset for offset, cities in {
        -5: "Atlanta|Boston|Brooklyn|Charlotte|Cleveland|Detroit|Indianapolis|Miami|New York|Orlando|Philadelphia|Toronto|Washington|Auburn Hills|Tampa",
        -6: "Chicago|Dallas|Houston|Memphis|Milwaukee|Minneapolis|New Orleans|Oklahoma City|San Antonio",
        -7: "Denver|Phoenix|Salt Lake City",
        -8: "Los Angeles|Inglewood|Oakland|San Francisco|Sacramento|Portland|Las Vegas",
    }.items() for city in cities.split("|")
}


def records(folder, table, columns):
    return duckdb.read_parquet(str(folder / f"{table}.parquet")).project(columns).fetchall()


def convert(folder):
    venues = {}
    if (folder / "espn-schedule.parquet").exists():
        for day, home, away, venue, start, city in records(
            folder, "espn-schedule",
            "game_date, home_id, away_id, venue_id, date, venue_address_city"
        ):
            if home and away and 1 <= home <= 30 and 1 <= away <= 30 and venue:
                key = (day, f"161061{ESPN_TEAMS[home-1]}", f"161061{ESPN_TEAMS[away-1]}")
                venues[key] = (str(venue), start, VENUE_ZONES.get((city or "").strip()))
    officials = defaultdict(set)
    if (folder / "nba-officials.parquet").exists():
        for game, official in records(folder, "nba-officials", "game_id, official_id"):
            if official is not None:
                officials[game].add(str(official))
    shared = defaultdict(lambda: defaultdict(list))
    stints_path = folder / "game_rotations.parquet"
    if stints_path.exists():
        stints = duckdb.read_parquet(str(stints_path))
        overlap = stints.query("stints", """
            SELECT a.game_id, a.team_id, a.player_id, b.player_id,
                   sum(least(a.out_time,b.out_time)-greatest(a.in_time,b.in_time))/60
            FROM stints a JOIN stints b
              ON a.game_id=b.game_id AND a.team_id=b.team_id AND a.player_id<b.player_id
             AND a.in_time<b.out_time AND b.in_time<a.out_time
            WHERE a.matches_box_score AND b.matches_box_score
            GROUP BY ALL
        """)
        for game, team, first, second, minutes in overlap.fetchall():
            shared[game][team].append([first, second, round(minutes, 3)])
    shots_by_game = {}
    events_path = folder / "game_events.parquet"
    if events_path.exists():
        events = duckdb.read_parquet(str(events_path))
        profiles = events.query("events", """
            SELECT game_id, team_id, count(*) FILTER (WHERE shot_distance <= 4),
                   count(*) FILTER (WHERE shot_value = 3 AND abs(x) >= 220 AND y <= 90),
                   count(*) FILTER (WHERE shot_value = 3), count(*)
            FROM events WHERE shot_value IN (2,3) AND shot_distance IS NOT NULL AND team_id IS NOT NULL
            GROUP BY ALL
        """)
        for game, team, rim, corner, three, attempts in profiles.fetchall():
            shots_by_game[game, team] = [rim / attempts, corner / attempts, three / attempts]
    starters = defaultdict(list)
    if (folder / "player_game_status.parquet").exists():
        for game, team, player, starter in records(
            folder, "player_game_status", "game_id, team_id, player_id, starter"
        ):
            if starter:
                starters[game, team].append(player)
    elif (folder / "game_rotations.parquet").exists():
        opening = duckdb.read_parquet(str(folder / "game_rotations.parquet")).filter(
            "in_time = 0 AND out_time > 0 AND matches_box_score"
        )
        for game, team, player in opening.project("game_id, team_id, player_id").fetchall():
            starters[game, team].append(player)
    scores = {(game, team): (points, pace, offense, defense)
              for game, team, points, pace, offense, defense in records(
        folder, "team_game_logs", "game_id, team_id, pts, pace, off_rating, def_rating"
    )}
    factors = {(game, team): [efg, turnovers, rebounds, fta / fga if fga else None, minutes, fouls, fta]
               for game, team, efg, turnovers, rebounds, fta, fga, minutes, fouls in records(
        folder, "team_game_logs", "game_id, team_id, efg_pct, tm_tov_pct, oreb_pct, fta, fga, min, pf"
    )}
    odds_path = folder / "betting_lines.parquet"
    totals = defaultdict(list)
    priced_totals = {}
    if odds_path.exists():
        for game, provider, line, over, under in records(
            folder, "betting_lines", "game_id, provider_id, close_total, close_over_odds, close_under_odds"
        ):
            if line is not None and 100 < line < 350:
                totals[game].append((provider, line))
                if provider == "100" and all(price is not None and abs(price) >= 100 for price in (over, under)):
                    priced_totals[game] = [line, over, under]
    boxes = defaultdict(dict)
    for game, player, team, minutes, points, rebounds, assists, shots, free_throws in records(
        folder, "player_game_logs", "game_id, player_id, team_id, min, pts, reb, ast, fga, fta"
    ):
        boxes[game][player] = dict(minutes=float(minutes), points=points,
                                  rebounds=rebounds, assists=assists,
                                  shots=shots or 0, free_throws=free_throws or 0, team=team)
    injury_snapshots = defaultdict(list)
    injury_path = folder / "injuries.parquet"
    if injury_path.exists():
        injury_rows = duckdb.read_parquet(str(injury_path)).query("injuries", """
            SELECT game_id, team_id, person_id, lower(current_status),
                   epoch(timezone('America/New_York', report_datetime))
            FROM injuries WHERE game_id IS NOT NULL AND team_id IS NOT NULL
            ORDER BY report_datetime DESC
        """)
        for game, team, player, status, stamp in injury_rows.fetchall():
            injury_snapshots[game, str(team)].append((stamp, player, status))
    for game, day, season, kind, home, away, neutral, round_number, series in records(
        folder, "games", "game_id, game_date, season, game_type, home_team_id, away_team_id, neutral_site, playoff_round, series_number"
    ):
        phase = {"regular_season": "regular", "play_in": "play_in", "playoffs": "playoffs"}.get(kind)
        if not phase or home is None or away is None:
            continue
        lineups = [sorted(set(starters[game, team])) for team in (home, away)]
        history_only = any(len(lineup) != 5 for lineup in lineups)
        if not boxes[game]:
            raise ValueError(f"{game} has no player box scores.")
        team_stats = [scores[game, team] for team in (home, away)]
        venue, start, zone = venues.get((day, home, away), (None, None, None))
        cutoff = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
        if cutoff is not None and cutoff.utcoffset() is None:
            cutoff = None
        availability = []
        if cutoff:
            for team in (home, away):
                prior = (row for row in injury_snapshots[game, team] if row[0] < cutoff.timestamp())
                first = next(prior, None)
                if first:
                    snapshot = datetime.fromtimestamp(first[0], timezone.utc).isoformat()
                    for stamp, player, status in [first, *prior]:
                        if stamp != first[0]:
                            break
                        if player is not None and status in ("available", "probable", "questionable", "doubtful", "out"):
                            availability.append(dict(team_id=team, player_id=str(player),
                                                     status=status, reported_at=snapshot))
        market = min(totals[game], key=lambda quote: (quote[0] != "odds_api_consensus", quote),
                     default=(None, None))
        yield dict(id=game, date=day.isoformat(), season=str(season - 1), phase=phase,
                   neutral=neutral, score=[row[0] for row in team_stats],
                   team_box=[row[1:] for row in team_stats],
                   factors=[factors[game, team] for team in (home, away)],
                   officials=sorted(officials[game]),
                   venue=venue, pregame_cutoff=cutoff.isoformat() if cutoff else None,
                   availability=availability,
                   venue_standard_utc_offset=zone,
                   shot_profile=[shots_by_game.get((game, team)) for team in (home, away)],
                   pair_minutes=[shared[game].get(team, []) for team in (home, away)],
                   series=[round_number, series] if phase == "playoffs" else None,
                   close_total=market[1], total_source=market[0],
                   draftkings_total=priced_totals.get(game),
                   history_only=history_only,
                   home=dict(team=home, roster=[] if history_only else lineups[0],
                             starters=None if history_only else lineups[0]),
                   away=dict(team=away, roster=[] if history_only else lineups[1],
                             starters=None if history_only else lineups[1]),
                   box=boxes[game])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="+", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    games = sorted((game for folder in args.folder for game in convert(folder)),
                   key=lambda game: (game["date"], game["id"]))
    if len({game["id"] for game in games}) != len(games):
        sys.exit("Duplicate game IDs across input folders.")
    with args.output.open("w", encoding="utf-8") as output:
        for game in games:
            output.write(json.dumps(game) + "\n")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["game_id", "date", "phase", "home_team", "away_team",
                         "home_starters", "away_starters", "home_points", "away_points",
                         "close_total", "total_source", "history_only"])
        for game in games:
            home, away = game["home"], game["away"]
            writer.writerow([game["id"], game["date"], game["phase"], home["team"], away["team"],
                             " ".join(home["starters"] or []), " ".join(away["starters"] or []),
                             *game["score"], game["close_total"], game["total_source"],
                             game["history_only"]])
    history = sum(game["history_only"] for game in games)
    print(f"Converted {len(games)} games ({history} history-only).")


if __name__ == "__main__":
    main()

