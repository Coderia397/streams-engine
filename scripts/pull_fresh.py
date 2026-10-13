#!/usr/bin/env python3
"""Pulls what is new and what is coming from TMDB into one file, `fresh.json`, for the engine's New & Popular pages: now playing, upcoming,
on the air, airing today, trending (day, week) and the most popular titles released in the last 90 days or in the next 180 days.
Each entry keeps the TMDB list fields the pages need plus the release date and the lists it came from.
  TMDB_KEY=... python3 scripts/pull_fresh.py [out.json]
Standard library only. The key is read from the environment and never printed."""
import datetime, json, os, sys, time, urllib.parse, urllib.request

KEY = os.environ.get("TMDB_KEY", "")
BASE = "https://api.themoviedb.org/3"
if not KEY:
    sys.exit("TMDB_KEY is not set")

def get(path, **params):
    q = urllib.parse.urlencode({"api_key": KEY, "language": "en-US", **params})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{BASE}{path}?{q}", headers={"User-Agent": "streams-fresh"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # network or rate limit: back off and retry
            err = type(e).__name__
            time.sleep(1.5 * (attempt + 1))
    sys.exit(f"{path} failed ({err})")

today = datetime.date.today()
past = (today - datetime.timedelta(days=90)).isoformat()
soon = (today + datetime.timedelta(days=180)).isoformat()

genres = {}
for kind in ("movie", "tv"):
    for g in get(f"/genre/{kind}/list")["genres"]:
        genres[g["id"]] = g["name"]

LISTS = [
    ("now_playing", "movie", "/movie/now_playing", {}, 5),
    ("upcoming", "movie", "/movie/upcoming", {}, 5),
    ("on_the_air", "tv", "/tv/on_the_air", {}, 5),
    ("airing_today", "tv", "/tv/airing_today", {}, 3),
    ("trending_day", None, "/trending/all/day", {}, 3),
    ("trending_week", None, "/trending/all/week", {}, 3),
    ("new_movie", "movie", "/discover/movie", {"sort_by": "popularity.desc", "primary_release_date.gte": past, "primary_release_date.lte": today.isoformat()}, 5),
    ("new_tv", "tv", "/discover/tv", {"sort_by": "popularity.desc", "first_air_date.gte": past, "first_air_date.lte": today.isoformat()}, 5),
    ("soon_movie", "movie", "/discover/movie", {"sort_by": "popularity.desc", "primary_release_date.gte": (today + datetime.timedelta(days=1)).isoformat(), "primary_release_date.lte": soon}, 5),
    ("soon_tv", "tv", "/discover/tv", {"sort_by": "popularity.desc", "first_air_date.gte": (today + datetime.timedelta(days=1)).isoformat(), "first_air_date.lte": soon}, 3),
]

items = {}
for tag, kind, path, params, pages in LISTS:
    position = 0
    for page in range(1, pages + 1):
        data = get(path, page=page, **params)
        for r in data.get("results", []):
            mt = kind or r.get("media_type")
            if mt not in ("movie", "tv"):
                continue
            date = r.get("release_date") if mt == "movie" else r.get("first_air_date")
            key = (mt, r["id"])
            e = items.setdefault(key, {
                "media_type": mt, "id": r["id"], "title": r.get("title") or r.get("name") or "", "original_language": r.get("original_language"),
                "overview": r.get("overview") or "", "poster_path": r.get("poster_path"), "backdrop_path": r.get("backdrop_path"),
                "genres": [genres[g] for g in r.get("genre_ids", []) if g in genres], "vote_average": r.get("vote_average") or 0,
                "vote_count": r.get("vote_count") or 0, "popularity": r.get("popularity") or 0, "release_date": date or "",
                "origin_country": r.get("origin_country") or [], "adult": bool(r.get("adult")), "lists": [], "ranks": {},
            })
            if tag not in e["lists"]:
                e["lists"].append(tag)
                e["ranks"][tag] = position  # place in that list, 0 first
            position += 1
        if page >= data.get("total_pages", 1):
            break

out = sys.argv[1] if len(sys.argv) > 1 else "fresh.json"
rows = sorted(items.values(), key=lambda e: (e["media_type"], e["id"]))
with open(out, "w", encoding="utf-8") as f:
    json.dump({"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "items": rows}, f, ensure_ascii=True, separators=(",", ":"))
dated = sum(1 for e in rows if e["release_date"])
print(f"{len(rows)} items, {dated} with a release date, written to {out}")
