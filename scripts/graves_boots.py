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
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
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

ALL_PLATFORMS = ["na1", "br1", "la1", "la2", "euw1", "eun1", "tr1", "ru", "me1",
                 "kr", "jp1", "oc1", "sg2", "tw2", "vn2"]
SERVER_NAMES = {"na1": "NA", "br1": "BR", "la1": "LAN", "la2": "LAS", "euw1": "EUW", "eun1": "EUNE",
                "tr1": "TR", "ru": "RU", "me1": "ME", "kr": "KR", "jp1": "JP", "oc1": "OCE",
                "sg2": "SEA", "tw2": "TW", "vn2": "VN", "ph2": "PH", "th2": "TH"}

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
        self.lock = threading.Lock()  # several platform threads share one regional host

    def wait(self) -> None:
        while True:
            with self.lock:
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


class LimiterRegistry:
    """One limiter per host. Riot app rate limits apply per routing value (na1, euw1,
    americas, europe, ...), so different hosts run fully in parallel."""

    def __init__(self, limits: list[tuple[int, float]]):
        self.limits = limits
        self.by_host: dict[str, RateLimiter] = {}
        self.lock = threading.Lock()

    def get(self, host: str) -> RateLimiter:
        with self.lock:
            if host not in self.by_host:
                self.by_host[host] = RateLimiter(self.limits)
            return self.by_host[host]


class AuthError(RuntimeError):
    pass


class RiotClient:
    def __init__(self, key: str, platform: str, region: str, limiters: LimiterRegistry,
                 cache_dir: Path):
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = key
        self.platform = platform
        self.platform_host = f"https://{platform}.api.riotgames.com"
        self.region_host = f"https://{region}.api.riotgames.com"
        self.limiters = {self.platform_host: limiters.get(self.platform_host),
                         self.region_host: limiters.get(self.region_host)}
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0

    def _get(self, host: str, path: str, params: dict | None = None, allow_404: bool = False):
        url = host + path
        for attempt in range(8):
            self.limiters[host].wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.ProxyError as exc:
                # The network proxy refused the host: not allowed, or no key saved for it.
                raise AuthError(f"{host} is blocked by the network proxy ({exc})") from exc
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
                print(f"  [{self.platform}] rate limited (429) on {host}; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (401, 403):
                raise AuthError(f"Riot API returned {resp.status_code} for {host}{path}. "
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

def is_legendary(item: dict) -> bool:
    """A completed item: buyable on Summoner's Rift, builds into nothing, not boots,
    consumable or trinket, and costs at least 1500 gold (which leaves out starters)."""
    tags = set(item.get("tags", []))
    gold = item.get("gold", {})
    return (item.get("maps", {}).get("11", False) and not item.get("into")
            and gold.get("purchasable", False) and gold.get("total", 0) >= 1500
            and not tags & {"Boots", "Consumable", "Trinket"})


def load_items(cache_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    """(tier-2 boots, completed items) by id. Tier-2 boots are the Summoner's Rift items
    tagged Boots that build straight from Boots (1001)."""
    try:
        version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=15).json()[0]
        cache = cache_dir / f"items_{version}.json"
        if cache.exists():
            items = json.loads(cache.read_text())
        else:
            items = requests.get(
                f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json", timeout=30
            ).json()["data"]
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(items))
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        cached = sorted(cache_dir.glob("items_*.json"))
        if not cached:
            sys.exit(f"Could not load Data Dragon item data ({exc}) and no cached copy exists.")
        version, items = cached[-1].stem.removeprefix("items_"), json.loads(cached[-1].read_text())
    boots = {
        int(item_id): item["name"]
        for item_id, item in items.items()
        if "Boots" in item.get("tags", [])
        and "1001" in item.get("from", [])
        and item.get("maps", {}).get("11", False)
    }
    # Keep ids from older patches too, in case cached games predate an item change.
    for item_id, name in FALLBACK_TIER2_BOOTS.items():
        boots.setdefault(item_id, name)
    legendaries = {int(i): item["name"] for i, item in items.items() if is_legendary(item)}
    print(f"Data Dragon {version}: {len(boots)} tier-2 boots, {len(legendaries)} completed items")
    return boots, legendaries


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def find_graves_players(client: RiotClient, tiers: list[str], max_checks: int, want: int,
                        recent_days: int) -> list[dict]:
    tag = f"[{client.platform}]"
    ladder = []
    for tier in tiers:
        league = client.apex_league(tier)
        entries = league.get("entries", [])
        print(f"{tag} {tier.title()}: {len(entries)} players")
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
            candidates.append({"puuid": puuid, "points": mastery.get("championPoints", 0),
                               "last_play": mastery.get("lastPlayTime", 0)})
        if i % 100 == 0 or i == len(ladder):
            print(f"{tag} mastery checked {i}/{len(ladder)} -> {len(candidates)} recent Graves players")

    # Most recently active on Graves first: lifetime points say little about current play.
    candidates.sort(key=lambda c: (c["last_play"], c["points"]), reverse=True)
    return candidates[:want]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def build_order(timeline: dict, participant_id: int, boots: dict[int, str],
                legendaries: dict[int, str]) -> dict:
    """Walk one player's purchases in order (undos removed) and say where tier-2 boots
    fall among completed items. boots_slot 1 = boots before any completed item,
    2 = boots right after the first completed item, and so on."""
    kept: list[tuple[int, int]] = []  # (timestamp, item_id) of tracked purchases, in order
    for frame in timeline["info"]["frames"]:
        for ev in frame.get("events", []):
            if ev.get("participantId") != participant_id:
                continue
            etype = ev.get("type")
            if etype == "ITEM_PURCHASED" and (ev.get("itemId") in boots or ev.get("itemId") in legendaries):
                kept.append((ev["timestamp"], ev["itemId"]))
            elif etype == "ITEM_UNDO":
                # An undo of a purchase: drop the most recent matching purchase.
                for j in range(len(kept) - 1, -1, -1):
                    if kept[j][1] == ev.get("beforeId"):
                        kept.pop(j)
                        break
    completed_before: list[str] = []
    seen_legendary: set[int] = set()
    boots_at = None
    for ts, item_id in kept:
        if item_id in boots:
            boots_at = (ts, item_id)
            break
        if item_id not in seen_legendary:  # rebuying the same item (e.g. after selling) isn't a new slot
            seen_legendary.add(item_id)
            completed_before.append(legendaries[item_id])
    total_completed = len({i for _, i in kept if i in legendaries})
    if boots_at is None:
        return {"boots_minute": None, "boots_item": "None", "boots_slot": None,
                "boots_position": "Not built", "items_before": "", "completed_items": total_completed}
    slot = len(completed_before) + 1
    return {
        "boots_minute": round(boots_at[0] / 60000, 2),
        "boots_item": boots.get(boots_at[1], str(boots_at[1])),
        "boots_slot": slot,
        "boots_position": f"{ordinal(slot)} item",
        "items_before": " > ".join(completed_before),
        "completed_items": total_completed,
    }


def graves_row(client: "RiotClient | None", mid: str, boots: dict[int, str], legendaries: dict[int, str],
               cache_dir: Path | None = None) -> list[dict]:
    """Rows for every Graves in one match (empty if none, or a remake). With no client,
    reads the match and timeline from cache_dir only."""
    if client is None:
        info = json.loads((cache_dir / "match" / f"{mid}.json").read_text())["info"]
    else:
        info = client.match(mid)["info"]
    if info.get("queueId") != RANKED_SOLO_QUEUE:
        return []
    duration_s = info.get("gameDuration", 0)
    # Before patch 11.20 gameDuration was in ms.
    if "gameEndTimestamp" not in info:
        duration_s //= 1000
    graves = [p for p in info["participants"] if p.get("championId") == GRAVES_ID]
    if not graves:
        return []
    # Skip remakes.
    if duration_s < 300 or any(p.get("gameEndedInEarlySurrender") for p in info["participants"]):
        return []
    if client is None:
        tl_path = cache_dir / "timeline" / f"{mid}.json"
        if not tl_path.exists():
            return []
        timeline = json.loads(tl_path.read_text())
    else:
        timeline = client.timeline(mid)
    rows = []
    for g in graves:
        order = build_order(timeline, g["participantId"], boots, legendaries)
        rows.append({
            "match_id": mid,
            "platform": str(info.get("platformId", mid.split("_")[0])).lower(),
            "game_version": info.get("gameVersion", ""),
            "game_start": info.get("gameStartTimestamp", 0),
            "puuid": g["puuid"],
            "riot_id": f"{g.get('riotIdGameName', '')}#{g.get('riotIdTagline', '')}",
            "position": g.get("teamPosition", ""),
            "win": bool(g["win"]),
            "game_minutes": round(duration_s / 60, 2),
            **order,
        })
    return rows


def collect(client: RiotClient, players: list[dict], matches_per_player: int, days: int,
            boots: dict[int, str], legendaries: dict[int, str], max_games: int,
            give_up_after: int) -> list[dict]:
    """Scan each seed player's recent ranked games. A seed with no Graves game in their
    first `give_up_after` matches is skipped, since they aren't playing him now."""
    start_time = int(time.time() - days * 86400) if days else None
    seen: set[str] = set()
    rows: list[dict] = []
    scanned = 0
    for n, p in enumerate(players, 1):
        if max_games and len(rows) >= max_games:
            break
        seed_graves = 0
        ids = client.match_ids(p["puuid"], matches_per_player, start_time)
        for k, mid in enumerate(ids, 1):
            if give_up_after and k > give_up_after and seed_graves == 0:
                break
            if mid in seen:
                # Already counted; still note whether this seed was the Graves.
                seed_graves += any(r["match_id"] == mid and r["puuid"] == p["puuid"] for r in rows)
                continue
            seen.add(mid)
            scanned += 1
            new_rows = graves_row(client, mid, boots, legendaries)
            rows.extend(new_rows)
            seed_graves += any(r["puuid"] == p["puuid"] for r in new_rows)
        print(f"[{client.platform}] player {n}/{len(players)}: {seed_graves} own Graves games | total {len(rows)} Graves games, "
              f"{scanned} matches scanned, {client.calls_made} API calls")
    return rows


def collect_from_cache(cache_root: Path, platforms: list[str], boots: dict[int, str],
                       legendaries: dict[int, str], days: int) -> list[dict]:
    """Every cached ranked match with a Graves and a cached timeline, from the given
    platforms and (if days) the last `days` days. No API calls."""
    cutoff_ms = (time.time() - days * 86400) * 1000 if days else 0
    rows = []
    for platform in platforms:
        cache_dir = cache_root / platform
        for path in sorted((cache_dir / "match").glob("*.json")):
            for r in graves_row(None, path.stem, boots, legendaries, cache_dir):
                if r["game_start"] >= cutoff_ms:
                    rows.append(r)
    by_platform = defaultdict(int)
    for r in rows:
        by_platform[r["platform"]] += 1
    print(f"{len(rows)} Graves games in the cache: " +
          ", ".join(f"{SERVER_NAMES.get(p, p)} {n}" for p, n in sorted(by_platform.items(), key=lambda kv: -kv[1])))
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
    out.append("== Tier-2 boots as which item (completed items before boots + 1) ==")
    out.append(header)
    slots: dict[int, list[dict]] = defaultdict(list)
    for r in built:
        slots[r["boots_slot"]].append(r)
    for slot in sorted(slots):
        grp = slots[slot]
        times = [g["boots_minute"] for g in grp]
        out.append(fmt_row(grp[0]["boots_position"], len(grp), sum(g["win"] for g in grp), total)
                   + f"  median {statistics.median(times):.1f} min")
    if unbuilt:
        out.append(fmt_row("not built", len(unbuilt), sum(g["win"] for g in unbuilt), total))

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
            "server": SERVER_NAMES.get(r.get("platform", ""), r.get("platform", "").upper()),
            "patch": major_minor,
            "date": time.strftime("%Y-%m-%d", time.gmtime(r["game_start"] / 1000)) if r["game_start"] else None,
            "player": player,
            "position": r["position"] or "UNKNOWN",
            "win": r["win"],
            "game_minutes": r["game_minutes"],
            "boots_minute": r["boots_minute"],
            "boots_item": r["boots_item"],
            "boots_slot": r["boots_slot"],
            "boots_position": r["boots_position"],
            "items_before": r["items_before"],
            "completed_items": r["completed_items"],
        })
    meta = [
        {"label": "graves_boots.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
        {"label": "Input", "value": f"Riot API ({platform}): apex ladder, Graves mastery, match-v5 matches and timelines"},
        {"label": "Servers", "value": ", ".join(sorted({SERVER_NAMES.get(r.get("platform", ""), r.get("platform", "")) for r in rows}))},
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
    ap.add_argument("--platforms", "--platform", default="na1",
                    help="comma-separated platforms (na1,euw1,kr,...) or 'all' for every server (default na1)")
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
    ap.add_argument("--give-up-after", type=int, default=10,
                    help="skip a seed player with no Graves game in their first N matches (0 = never)")
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many Graves games (0 = no limit)")
    ap.add_argument("--bucket", type=int, default=1, help="bucket width in minutes for the distribution")
    ap.add_argument("--rate", default="20/1,100/120",
                    help="app rate limits as count/seconds pairs (dev key default: 20/1,100/120)")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / ".riot_cache"))
    ap.add_argument("--out", default="graves_boots", help="output prefix for the .csv, .json and .txt files")
    ap.add_argument("--from-cache", action="store_true",
                    help="no API calls: analyse every Graves game already in the cache")
    args = ap.parse_args()

    platforms = ALL_PLATFORMS if args.platforms.lower() == "all" else \
        [p.strip().lower() for p in args.platforms.split(",") if p.strip()]
    unknown = [p for p in platforms if p not in PLATFORM_TO_REGION]
    if unknown:
        sys.exit(f"Unknown platform(s) {unknown}. Known: {', '.join(PLATFORM_TO_REGION)}")
    limits = [(int(c), float(s)) for c, s in (pair.split("/") for pair in args.rate.split(","))]
    cache_root = Path(args.cache_dir)

    boots, legendaries = load_items(cache_root)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    calls = 0
    if not args.from_cache:
        key = load_key(args.key)
        registry = LimiterRegistry(limits)

        def run_platform(platform: str) -> tuple[str, int, int]:
            client = RiotClient(key, platform, PLATFORM_TO_REGION[platform], registry, cache_root / platform)
            players = find_graves_players(client, tiers, args.max_mastery_checks, args.graves_players,
                                          args.recent_days)
            print(f"[{platform}] using {len(players)} recent Graves players")
            found = collect(client, players, args.matches_per_player, args.days, boots, legendaries,
                            args.max_games, args.give_up_after) if players else []
            return platform, len(found), client.calls_made

        # One thread per platform. Platforms in the same region share that region's
        # match-v5 limiter; every other host runs at its own full rate.
        failed = []
        with ThreadPoolExecutor(max_workers=len(platforms)) as pool:
            futures = {pool.submit(run_platform, p): p for p in platforms}
            for fut in as_completed(futures):
                platform = futures[fut]
                try:
                    _, n, c = fut.result()
                    calls += c
                    print(f"[{platform}] done: {n} Graves games, {c} API calls")
                except Exception as exc:  # one server failing shouldn't stop the others
                    failed.append(platform)
                    print(f"[{platform}] FAILED: {exc}", file=sys.stderr)
        if failed:
            print(f"Servers that failed: {', '.join(failed)}", file=sys.stderr)

    # The output is everything cached for these servers in the window, so games from
    # earlier runs count too.
    rows = collect_from_cache(cache_root, platforms, boots, legendaries, args.days)
    if not rows:
        sys.exit("No Graves games found.")
    platform = ",".join(platforms)

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
    print(f"\nPer-game rows: {csv_path}   report: {args.out}.txt   API calls: {calls}")


if __name__ == "__main__":
    main()
