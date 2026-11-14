"""Summarize prior NBA games for a pregame matchup."""

from collections import defaultdict, deque
from datetime import datetime
from itertools import combinations
from math import log1p


AVAILABILITY = {"available": 1., "probable": .75, "questionable": .5,
                "doubtful": .25, "out": 0., "suspended": 0.}


def pregame_availability(game):
    reports = game.get("availability", [])
    if not reports:
        return {}
    cutoff = datetime.fromisoformat(game["pregame_cutoff"].replace("Z", "+00:00"))
    if cutoff.utcoffset() is None:
        raise ValueError("Pregame cutoff requires a UTC offset.")
    latest = {}
    for report in reports:
        stamp = datetime.fromisoformat(report["reported_at"].replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError("Availability reports require a UTC offset.")
        if stamp >= cutoff:
            continue
        key = (report["team_id"], report["player_id"])
        status = AVAILABILITY[report["status"]]
        if key not in latest or stamp > latest[key][0]:
            latest[key] = (stamp, status)
    return {key: record[1] for key, record in latest.items()}


class MatchupHistory:
    def __init__(self):
        self.team = defaultdict(lambda: deque(maxlen=20))
        self.pairs = defaultdict(float)
        self.series = defaultdict(int)
        self.zones = {}

    def features(self, game, day, side):
        own, other = (game[key]["team"] for key in ("home", "away"))
        if side:
            own, other = other, own
        history = self.team[own]
        opponent = self.team[other]
        workload = [sum((day - row[0]).days <= window for row in history) / window
                    for window in (4, 7, 14)]
        road_run = next((i for i, row in enumerate(reversed(history)) if not row[1]), len(history))
        metrics = []
        for i, (center, scale) in enumerate(zip(
                (.52, .14, .25, .24, 240, .3, .08, .38, 0),
                (.1, .1, .15, .2, 25, .2, .1, .2, 20))):
            attack = [row[2][i] for row in history if row[2][i] is not None]
            allowed = [row[3][i] for row in opponent if row[3][i] is not None]
            for values in (attack, allowed):
                metrics.extend(((sum(values) / len(values) - center) / scale if values else 0,
                                len(values) / 20))
        starters = game["away" if side else "home"].get("starters") or ()
        minutes = sorted((self.pairs.get((own, first, second), 0.)
                          for first, second in combinations(sorted(starters), 2)), reverse=True)
        series = self.series_key(game)
        zone = game.get("venue_standard_utc_offset")
        previous = self.zones.get(own)
        known = zone is not None and previous is not None
        shift = zone - previous if known else 0
        return [*workload, road_run / 20, *metrics,
                *([log1p(value) / 8 for value in minutes] + [0.] * (10 - len(minutes))),
                self.series.get((series, own), 0) / 4,
                self.series.get((series, other), 0) / 4,
                shift / 3, abs(shift) / 3, float(known)]

    @staticmethod
    def series_key(game):
        series = game.get("series")
        if series and all(value is not None for value in series):
            return game.get("season"), *series
        return None

    def update(self, game, day, adjusted_margin):
        factors = game.get("factors") or [None, None]
        shots = game.get("shot_profile") or [None, None]
        rows = [list((factors[i] or [None] * 5)[:5]) + list(shots[i] or [None] * 3)
                + [adjusted_margin[i]] for i in range(2)]
        series = self.series_key(game)
        for side, label in enumerate(("home", "away")):
            team = game[label]["team"]
            self.zones[team] = game.get("venue_standard_utc_offset")
            self.team[team].append((day, side == 1 and not game.get("neutral"), rows[side], rows[1 - side]))
            for first, second, minutes in game.get("pair_minutes", ([], []))[side]:
                self.pairs[team, first, second] += minutes
            if series is not None and game["score"][side] > game["score"][1 - side]:
                self.series[series, team] += 1

