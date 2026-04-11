"""
Adversarial / edge-case profile runner.
Run from the project root:  python src/adversarial_test.py
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from recommender import load_songs, recommend_songs, score_song

_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
songs = load_songs(_CSV)

PROFILES = [
    (
        "Profile 1 — Conflicting Signals (energy=0.9, mood=sad)",
        {
            "favorite_genre": "blues",
            "favorite_mood":  "sad",
            "target_energy":  0.9,
            "likes_acoustic": True,   # silently ignored
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
    (
        "Profile 2 — Strict Filter Blackhole (classical + angry, both strict)",
        {
            "favorite_genre": "classical",
            "favorite_mood":  "angry",
            "target_energy":  0.5,
            "likes_acoustic": False,
            "strict_genre":   True,
            "strict_mood":    True,
        },
    ),
    (
        "Profile 3 — Mood Label Trap (relaxed ≠ chill, ambient genre)",
        {
            "favorite_genre": "ambient",
            "favorite_mood":  "relaxed",
            "target_energy":  0.3,
            "likes_acoustic": True,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
    (
        "Profile 4 — Energy Floor (target=0.0, classical peaceful)",
        {
            "favorite_genre": "classical",
            "favorite_mood":  "peaceful",
            "target_energy":  0.0,
            "likes_acoustic": True,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
    (
        "Profile 5 — Acousticness Ghost (likes_acoustic=False, folk melancholic)",
        {
            "favorite_genre": "folk",
            "favorite_mood":  "melancholic",
            "target_energy":  0.3,
            "likes_acoustic": False,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
    (
        "Profile 6 — Tie Ambush (lofi focused, energy=0.60)",
        {
            "favorite_genre": "lofi",
            "favorite_mood":  "focused",
            "target_energy":  0.60,
            "likes_acoustic": False,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
    (
        "Profile 7 — Energy Override (rock+sad, target=0.95)",
        {
            "favorite_genre": "rock",
            "favorite_mood":  "sad",
            "target_energy":  0.95,
            "likes_acoustic": False,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
    (
        "Profile 8 — Middle Energy Attractor (hip-hop angry, target=0.5)",
        {
            "favorite_genre": "hip-hop",
            "favorite_mood":  "angry",
            "target_energy":  0.5,
            "likes_acoustic": False,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    ),
]


def run_profile(label, user_prefs):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Genre: {user_prefs['favorite_genre']}  |  Mood: {user_prefs['favorite_mood']}  |  Energy: {user_prefs['target_energy']}")
    strict_parts = []
    if user_prefs.get("strict_genre"):
        strict_parts.append(f"genre STRICT")
    if user_prefs.get("strict_mood"):
        strict_parts.append(f"mood STRICT")
    if strict_parts:
        print(f"  Strict filters: {', '.join(strict_parts)}")
    print(f"{'='*60}")

    results = recommend_songs(user_prefs, songs, k=5)

    if not results:
        print("  *** EMPTY RESULT — no songs survived strict filters ***")
        return

    for rank, (song, score, _) in enumerate(results, start=1):
        _, reasons = score_song(user_prefs, song)
        mood_tag  = "[MOOD MATCH]"  if song["mood"]  == user_prefs["favorite_mood"]  else ""
        genre_tag = "[GENRE MATCH]" if song["genre"] == user_prefs["favorite_genre"] else ""
        tags = f" {mood_tag}{genre_tag}".rstrip()
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"       ({song['genre']}, {song['mood']}, energy={song['energy']}, acousticness={song['acousticness']}){tags}")
        print(f"       Score: {score:.4f} / 5.00")
        for r in reasons:
            print(f"         • {r}")
        print()


for label, prefs in PROFILES:
    run_profile(label, prefs)
