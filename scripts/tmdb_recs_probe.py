#!/usr/bin/env python3
"""One off check: how many good titles are in TMDB's own recommendations (made from what its users do) for the anchors of eval/recs_cases.json,
measured the way ai-engine/scripts/eval_recs.py measures the engine (hits of the first 10 against the hand written lists).
  TMDB_KEY=... python3 scripts/tmdb_recs_probe.py"""
import json, os, sys, urllib.request

KEY = os.environ["TMDB_KEY"]
cases = json.load(open(os.path.join(os.path.dirname(__file__), "..", "eval", "recs_cases.json")))
def get(path):
    with urllib.request.urlopen(f"https://api.themoviedb.org/3{path}{'&' if '?' in path else '?'}api_key={KEY}&language=en-US", timeout=30) as r:
        return json.load(r)
total = ok3 = 0
for c in cases:
    res = []
    for kind in ("recommendations", "similar"):
        titles = [(r.get("title") or r.get("name") or "") for r in get(f"/{c['type']}/{c['id']}/{kind}")["results"][:10]]
        hits = [t for t in titles if any(g.lower() == t.lower() or g.lower() in t.lower() for g in c["good"])]
        res.append((kind, len(hits), titles[:6]))
    r = res[0]
    total += r[1]; ok3 += r[1] >= 3
    print(f"{c['name']:22} recs {res[0][1]:2} similar {res[1][1]:2}  {res[0][2]}")
print(f"TMDB recommendations: mean {total / len(cases):.2f} hits of 10, {ok3}/{len(cases)} anchors with 3 or more")
