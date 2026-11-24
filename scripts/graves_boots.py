#!/usr/bin/env python3
"""
Graves tier-2 boots timing analysis (Riot API).

Finds ranked solo/duo games with a Graves in them, reads each match timeline,
works out when Graves finished his first tier-2 boots, and reports:
  * the distribution of tier-2 boots completion times
  * the winrate in each time bucket
  * the winrate by boot type, and for games where no tier-2 boots were built

How Graves games are found:
  1. Pull the apex ladder (Challenger / Grandmaster / Master) for the platform.
  2. Check each player's Graves mastery (champion-mastery-v4) and keep the
     players with the most Graves mastery who have played him recently.
  3. Pull those players' recent ranked match IDs and fetch each match.
     EVERY Graves in a fetched match is counted, not only the seed player.
  4. Fetch the timeline for each match that has a Graves in it.

Everything is cached under --cache-dir, so a re-run (or a run with bigger
limits) only fetches what it does not already have.

API key lookup order: --key, $RIOT_API_KEY, src/main/java/riotkey.txt.

Usage:
  python3 scripts/graves_boots.py
  python3 scripts/graves_boots.py --platform euw1 --graves-players 60 --matches-per-player 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import requests

GRAVES_ID = 104
GRAVES_NAME = "Graves"
RANKED_SOLO_QUEUE = 420

PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea", "ph2": "sea", "th2": "sea",
}

# Used only if Data Dragon cannot be reached. Tier-2 boots are the items built
# directly out of Boots (1001).
FALLBACK_TIER2_BOOTS = {
    3006: "Berserker's Greaves",
    3009: "Boots of Swiftness",
    3020: "Sorcerer's Shoes",
    3047: "Plated Steelcaps",
    3111: "Mercury's Treads",
    3117: "Mobility Boots",
    3158: "Ionian Boots of Lucidity",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY_FILE = REPO_ROOT / "src" / "main" / "java" / "riotkey.txt"


# --------------------------------------------------------------------------- #
# HTTP / rate limiting
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Sliding-window limiter supporting several windows, e.g. 20/1s + 100/120s."""

    def __init__(self, limits: list[tuple[int, float]]):
        self.limits = limits
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            longest = max(w for _, w in self.limits)
            while self.calls and now - self.calls[0] > longest:
                self.calls.popleft()
            sleep_for = 0.0
            for count, window in self.limits:
                in_window = [t for t in self.calls if now - t < window]
                if len(in_window) >= count:
                    sleep_for = max(sleep_for, window - (now - in_window[0]) + 0.05)
            if sleep_for <= 0:
                self.calls.append(now)
                return
            time.sleep(sleep_for)


class RiotClient:
    def __init__(self, key: str, platform: str, region: str, limits: list[tuple[int, float]],
                 cache_dir: Path):
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = key
        self.platform_host = f"https://{platform}.api.riotgames.com"
        self.region_host = f"https://{region}.api.riotgames.com"
        # Riot app rate limits are enforced per routing value, so one limiter per host.
        self.limiters = {self.platform_host: RateLimiter(limits), self.region_host: RateLimiter(limits)}
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0

    def _get(self, host: str, path: str, params: dict | None = None, allow_404: bool = False):
        url = host + path
        for attempt in range(8):
            self.limiters[host].wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                wait = 2 ** attempt
                print(f"  network error ({exc}); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            self.calls_made += 1
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "10"))
                print(f"  rate limited (429); sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (401, 403):
                sys.exit(f"Riot API returned {resp.status_code} for {path}. "
                         "The API key is missing, invalid, or expired (dev keys last 24h).")
            raise RuntimeError(f"GET {url} -> {resp.status_code}: {resp.text[:200]}")
        raise RuntimeError(f"GET {url} failed after retries")

    def _cached(self, name: str, fetch):
        path = self.cache_dir / name
        if path.exists():
            return json.loads(path.read_text())
        data = fetch()
        if data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return data

    # --- endpoints -------------------------------------------------------- #

    def apex_league(self, tier: str) -> dict:
        path = f"/lol/league/v4/{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5"
        return self._get(self.platform_host, path)

    def summoner_by_id(self, summoner_id: str) -> dict:
        return self._cached(f"summoner/{summoner_id}.json", lambda: self._get(
            self.platform_host, f"/lol/summoner/v4/summoners/{summoner_id}"))

    def graves_mastery(self, puuid: str) -> dict | None:
        path = f"/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{GRAVES_ID}"
        # Cache "no mastery" too, as an empty dict.
        data = self._cached(f"mastery/{puuid}.json",
                            lambda: self._get(self.platform_host, path, allow_404=True) or {})
        return data or None

    def match_ids(self, puuid: str, count: int, start_time: int | None) -> list[str]:
        params = {"queue": RANKED_SOLO_QUEUE, "type": "ranked", "start": 0, "count": count}
        if start_time:
            params["startTime"] = start_time
        return self._get(self.region_host, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params)

    def match(self, match_id: str) -> dict:
        return self._cached(f"match/{match_id}.json", lambda: self._get(
            self.region_host, f"/lol/match/v5/matches/{match_id}"))

    def timeline(self, match_id: str) -> dict:
        return self._cached(f"timeline/{match_id}.json", lambda: self._get(
            self.region_host, f"/lol/match/v5/matches/{match_id}/timeline"))


# --------------------------------------------------------------------------- #
# Item data
# --------------------------------------------------------------------------- #

def load_tier2_boots(cache_dir: Path) -> dict[int, str]:
    """Tier-2 boots = Summoner's Rift items tagged Boots that build from Boots (1001)."""
    try:
        version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=15).json()[0]
        cache = cache_dir / f"items_{version}.json"
        if cache.exists():
            items = json.loads(cache.read_text())
        else:
            items = requests.get(
                f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json", timeout=30
            ).json()["data"]
            cache.write_text(json.dumps(items))
        boots = {
            int(item_id): item["name"]
            for item_id, item in items.items()
            if "Boots" in item.get("tags", [])
            and "1001" in item.get("from", [])
            and item.get("maps", {}).get("11", False)
        }
        if boots:
            # Keep ids from older patches too, in case cached games predate an item change.
            for item_id, name in FALLBACK_TIER2_BOOTS.items():
                boots.setdefault(item_id, name)
            print(f"Tier-2 boots from Data Dragon {version}: {', '.join(sorted(boots.values()))}")
            return boots
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        print(f"Could not load Data Dragon ({exc}); using built-in boots list", file=sys.stderr)
    return dict(FALLBACK_TIER2_BOOTS)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def find_graves_players(client: RiotClient, tiers: list[str], max_checks: int, want: int,
                        recent_days: int) -> list[dict]:
    ladder = []
    for tier in tiers:
        league = client.apex_league(tier)
        entries = league.get("entries", [])
        print(f"{tier.title()}: {len(entries)} players")
        ladder.extend(entries)
    ladder.sort(key=lambda e: e.get("leaguePoints", 0), reverse=True)
    ladder = ladder[:max_checks]

    cutoff_ms = (time.time() - recent_days * 86400) * 1000
    candidates = []
    for i, entry in enumerate(ladder, 1):
        puuid = entry.get("puuid")
        if not puuid:  # older API responses only had summonerId
            puuid = client.summoner_by_id(entry["summonerId"])["puuid"]
        mastery = client.graves_mastery(puuid)
        if mastery and mastery.get("lastPlayTime", 0) >= cutoff_ms:
            candidates.append({"puuid": puuid, "points": mastery.get("championPoints", 0)})
        if i % 25 == 0 or i == len(ladder):
            print(f"  mastery checked {i}/{len(ladder)} -> {len(candidates)} recent Graves players")

    candidates.sort(key=lambda c: c["points"], reverse=True)
    return candidates[:want]


def first_tier2_boots(timeline: dict, participant_id: int, boots: dict[int, str]):
    """Return (timestamp_ms, item_id) of the first tier-2 boots kept, or None."""
    purchases: list[list] = []  # [timestamp, item_id], in order
    for frame in timeline["info"]["frames"]:
        for ev in frame.get("events", []):
            if ev.get("participantId") != participant_id:
                continue
            etype = ev.get("type")
            if etype == "ITEM_PURCHASED" and ev.get("itemId") in boots:
                purchases.append([ev["timestamp"], ev["itemId"]])
            elif etype == "ITEM_UNDO" and ev.get("beforeId") in boots:
                # Undo of a boots purchase: drop the most recent matching purchase.
                for j in range(len(purchases) - 1, -1, -1):
                    if purchases[j][1] == ev["beforeId"]:
                        purchases.pop(j)
                        break
    if not purchases:
        return None
    return purchases[0][0], purchases[0][1]


def collect(client: RiotClient, players: list[dict], matches_per_player: int, days: int,
            boots: dict[int, str], max_games: int) -> list[dict]:
    start_time = int(time.time() - days * 86400) if days else None
    match_ids: list[str] = []
    seen = set()
    for p in players:
        for mid in client.match_ids(p["puuid"], matches_per_player, start_time):
            if mid not in seen:
                seen.add(mid)
                match_ids.append(mid)
    print(f"{len(match_ids)} unique ranked matches to scan")

    rows = []
    for i, mid in enumerate(match_ids, 1):
        if max_games and len(rows) >= max_games:
            break
        if i % 50 == 0:
            print(f"  scanned {i}/{len(match_ids)} matches, {len(rows)} Graves games, "
                  f"{client.calls_made} API calls")
        match = client.match(mid)
        info = match["info"]
        if info.get("queueId") != RANKED_SOLO_QUEUE:
            continue
        duration_s = info.get("gameDuration", 0)
        # Before patch 11.20 gameDuration was in ms.
        if "gameEndTimestamp" not in info:
            duration_s //= 1000
        graves = [p for p in info["participants"] if p.get("championId") == GRAVES_ID]
        if not graves:
            continue
        # Skip remakes.
        if duration_s < 300 or any(p.get("gameEndedInEarlySurrender") for p in info["participants"]):
            continue
        timeline = client.timeline(mid)
        for g in graves:
            found = first_tier2_boots(timeline, g["participantId"], boots)
            rows.append({
                "match_id": mid,
                "game_version": info.get("gameVersion", ""),
                "game_start": info.get("gameStartTimestamp", 0),
                "puuid": g["puuid"],
                "riot_id": f"{g.get('riotIdGameName', '')}#{g.get('riotIdTagline', '')}",
                "position": g.get("teamPosition", ""),
                "win": bool(g["win"]),
                "game_minutes": round(duration_s / 60, 2),
                "boots_minute": round(found[0] / 60000, 2) if found else None,
                "boots_item": boots.get(found[1], str(found[1])) if found else "None",
            })
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def fmt_row(label: str, n: int, wins: int, total: int) -> str:
    wr = wins / n if n else 0
    lo, hi = wilson(wins, n)
    share = n / total if total else 0
    bar = "#" * round(share * 50)
    return (f"{label:>14} | {n:5d} | {share:6.1%} | {wr:6.1%} | "
            f"[{lo:5.1%} - {hi:5.1%}] | {bar}")


def report(rows: list[dict], bucket_minutes: int) -> str:
    out = []
    total = len(rows)
    wins = sum(r["win"] for r in rows)
    built = [r for r in rows if r["boots_minute"] is not None]
    unbuilt = [r for r in rows if r["boots_minute"] is None]
    out.append(f"Graves games analysed: {total}   overall winrate: {wins / total:.1%}")
    out.append(f"Unique Graves players: {len({r['puuid'] for r in rows})}")
    if built:
        times = sorted(r["boots_minute"] for r in built)
        q = statistics.quantiles(times, n=4) if len(times) >= 4 else [times[0], times[len(times) // 2], times[-1]]
        out.append(f"Tier-2 boots completed in {len(built)}/{total} games "
                   f"({len(built) / total:.1%}). Median {statistics.median(times):.1f} min, "
                   f"IQR {q[0]:.1f}-{q[2]:.1f} min, mean {statistics.mean(times):.1f} min")

    header = f"{'':>14} | {'games':>5} | {'share':>6} | {'WR':>6} | {'95% CI':^17} |"
    out.append("")
    out.append(f"== Tier-2 boots completion time ({bucket_minutes}-min buckets) ==")
    out.append(header)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in built:
        buckets[int(r["boots_minute"] // bucket_minutes)].append(r)
    for b in sorted(buckets):
        grp = buckets[b]
        label = f"{b * bucket_minutes}-{(b + 1) * bucket_minutes} min"
        out.append(fmt_row(label, len(grp), sum(g["win"] for g in grp), total))
    if unbuilt:
        out.append(fmt_row("never", len(unbuilt), sum(g["win"] for g in unbuilt), total))

    if built:
        times = sorted(r["boots_minute"] for r in built)
        median = statistics.median(times)
        early = [r for r in built if r["boots_minute"] <= median]
        late = [r for r in built if r["boots_minute"] > median]
        out.append("")
        out.append(f"== Split at the median ({median:.1f} min) ==")
        out.append(header)
        out.append(fmt_row("<= median", len(early), sum(r["win"] for r in early), total))
        out.append(fmt_row("> median", len(late), sum(r["win"] for r in late), total))

    out.append("")
    out.append("== By boots type ==")
    out.append(header)
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_item[r["boots_item"]].append(r)
    for item, grp in sorted(by_item.items(), key=lambda kv: -len(kv[1])):
        times = [g["boots_minute"] for g in grp if g["boots_minute"] is not None]
        label = item[:14]
        line = fmt_row(label, len(grp), sum(g["win"] for g in grp), total)
        if times:
            line += f"  median {statistics.median(times):.1f} min"
        out.append(line)

    out.append("")
    out.append("Note: winning teams earn gold faster, so earlier boots partly reflects being ahead "
               "rather than causing wins. Treat the time-vs-winrate trend as correlation.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #

def export_json(rows: list[dict], path: Path, platform: str) -> None:
    """Dashboard export: player identities replaced by P1, P2, ... and the script kept as meta."""
    ids: dict[str, str] = {}
    out = []
    for r in rows:
        player = ids.setdefault(r["puuid"], f"P{len(ids) + 1}")
        major_minor = ".".join(r["game_version"].split(".")[:2])
        out.append({
            "match_id": r["match_id"],
            "patch": major_minor,
            "date": time.strftime("%Y-%m-%d", time.gmtime(r["game_start"] / 1000)) if r["game_start"] else None,
            "player": player,
            "position": r["position"] or "UNKNOWN",
            "win": r["win"],
            "game_minutes": r["game_minutes"],
            "boots_minute": r["boots_minute"],
            "boots_item": r["boots_item"],
        })
    meta = [
        {"label": "graves_boots.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
        {"label": "Input", "value": f"Riot API ({platform}): apex ladder, Graves mastery, match-v5 matches and timelines"},
        {"label": "Pulled", "value": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())},
    ]
    path.write_text(json.dumps({"rows": out, "meta": meta}))


def load_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("RIOT_API_KEY")
    if not key and DEFAULT_KEY_FILE.exists():
        key = DEFAULT_KEY_FILE.read_text().strip()
    if not key:
        sys.exit("No Riot API key. Pass --key, set RIOT_API_KEY, or put it in src/main/java/riotkey.txt")
    return key.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", help="Riot API key (default: $RIOT_API_KEY or src/main/java/riotkey.txt)")
    ap.add_argument("--platform", default="na1", help="platform routing value (default na1)")
    ap.add_argument("--tiers", default="challenger,grandmaster,master",
                    help="apex tiers to seed players from")
    ap.add_argument("--max-mastery-checks", type=int, default=300,
                    help="how many ladder players (by LP) to check for Graves mastery")
    ap.add_argument("--graves-players", type=int, default=40,
                    help="how many top-mastery Graves players to pull matches for")
    ap.add_argument("--recent-days", type=int, default=30,
                    help="seed players must have played Graves within this many days")
    ap.add_argument("--matches-per-player", type=int, default=20, help="ranked match IDs per player (max 100)")
    ap.add_argument("--days", type=int, default=30, help="only matches from the last N days (0 = no limit)")
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many Graves games (0 = no limit)")
    ap.add_argument("--bucket", type=int, default=1, help="bucket width in minutes for the distribution")
    ap.add_argument("--rate", default="20/1,100/120",
                    help="app rate limits as count/seconds pairs (dev key default: 20/1,100/120)")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / ".riot_cache"))
    ap.add_argument("--out", default="graves_boots", help="output prefix for the .csv and .txt files")
    args = ap.parse_args()

    platform = args.platform.lower()
    region = PLATFORM_TO_REGION.get(platform)
    if not region:
        sys.exit(f"Unknown platform '{platform}'. Known: {', '.join(PLATFORM_TO_REGION)}")
    limits = [(int(c), float(s)) for c, s in (pair.split("/") for pair in args.rate.split(","))]
    cache_dir = Path(args.cache_dir) / platform

    client = RiotClient(load_key(args.key), platform, region, limits, cache_dir)
    boots = load_tier2_boots(cache_dir)

    players = find_graves_players(client, [t.strip() for t in args.tiers.split(",") if t.strip()],
                                  args.max_mastery_checks, args.graves_players, args.recent_days)
    if not players:
        sys.exit("No recent Graves players found; try more --max-mastery-checks or a larger --recent-days")
    print(f"Using {len(players)} Graves players (top mastery: {players[0]['points']:,} pts)")

    rows = collect(client, players, args.matches_per_player, args.days, boots, args.max_games)
    if not rows:
        sys.exit("No Graves games found in the scanned matches.")

    csv_path = Path(f"{args.out}.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    export_json(rows, Path(f"{args.out}.json"), platform)

    text = report(rows, args.bucket)
    Path(f"{args.out}.txt").write_text(text + "\n")
    print()
    print(text)
    print(f"\nPer-game rows: {csv_path}   report: {args.out}.txt   API calls: {client.calls_made}")


if __name__ == "__main__":
    main()
