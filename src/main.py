"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import os
from recommender import load_songs, recommend_songs, score_song

_HERE = os.path.dirname(__file__)
_CSV  = os.path.join(_HERE, "..", "data", "songs.csv")

VALID_GENRES = [
    "ambient", "blues", "classical", "country", "edm",
    "folk", "hip-hop", "indie pop", "jazz", "lofi",
    "metal", "pop", "r&b", "rock", "synthwave",
]
VALID_MOODS = [
    "angry", "chill", "confident", "energetic", "focused",
    "happy", "intense", "melancholic", "moody", "nostalgic",
    "peaceful", "relaxed", "romantic", "sad",
]


def ask_user_prefs() -> dict:
    """Collect preferences interactively from the terminal."""
    print("\n" + "="*50)
    print("  Tell us what you're in the mood for")
    print("="*50)

    # Genre
    print(f"\n  Available genres: {', '.join(VALID_GENRES)}")
    while True:
        genre = input("  Your favorite genre: ").strip().lower()
        if genre in VALID_GENRES:
            break
        print(f"  Not found. Pick one from the list above.")

    strict_genre = input("  Must songs match this genre exactly? (y/n): ").strip().lower() == "y"

    # Mood
    print(f"\n  Available moods: {', '.join(VALID_MOODS)}")
    while True:
        mood = input("  Your favorite mood: ").strip().lower()
        if mood in VALID_MOODS:
            break
        print(f"  Not found. Pick one from the list above.")

    strict_mood = input("  Must songs match this mood exactly? (y/n): ").strip().lower() == "y"

    # Energy
    print("\n  Energy scale: 0.0 = very calm, 1.0 = very intense")
    while True:
        try:
            energy = float(input("  Target energy (0.0 – 1.0): ").strip())
            if 0.0 <= energy <= 1.0:
                break
            print("  Please enter a number between 0.0 and 1.0.")
        except ValueError:
            print("  Please enter a valid number.")

    return {
        "favorite_genre":  genre,
        "favorite_mood":   mood,
        "target_energy":   energy,
        "likes_acoustic":  False,
        "strict_genre":    strict_genre,
        "strict_mood":     strict_mood,
    }


def print_recommendations(label: str, user_prefs: dict, songs: list, k: int = 3) -> None:
    strict_note = []
    if user_prefs.get("strict_genre"):
        strict_note.append(f"genre='{user_prefs['favorite_genre']}' (strict)")
    if user_prefs.get("strict_mood"):
        strict_note.append(f"mood='{user_prefs['favorite_mood']}' (strict)")

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Genre: {user_prefs['favorite_genre']}  |  Mood: {user_prefs['favorite_mood']}  |  Energy: {user_prefs['target_energy']}")
    if strict_note:
        print(f"  Strict filters: {', '.join(strict_note)}")
    print(f"{'='*50}")

    recommendations = recommend_songs(user_prefs, songs, k=k)

    if not recommendations:
        print("  No songs matched your strict filters. Try relaxing them.")
        return

    for rank, (song, score, _) in enumerate(recommendations, start=1):
        _, reasons = score_song(user_prefs, song)
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Score: {score:.2f} / 5.00")
        for reason in reasons:
            print(f"         • {reason}")
        print()


def main() -> None:
    songs = load_songs(_CSV)
    print(f"Loaded songs: {len(songs)}")

    # --- Three distinct user preference profiles ---

    # Profile A: wants high-energy pop to match a happy, upbeat mood
    # Tests whether mood + energy together surface the right pop songs
    # Expected top result: a pop song with high energy and happy mood

    # Profile B: wants calm lofi for studying or winding down
    # Tests the low-energy chill end of the catalog
    # Expected top result: a lofi song with low energy and chill mood

    # Profile C: wants the most intense, aggressive rock available
    # Tests the extreme high-energy end with a specific genre requirement
    # Expected top result: the highest-energy rock/metal song in the catalog
    profiles = {
        "High-Energy Pop": {
            "favorite_genre": "pop",
            "favorite_mood":  "happy",
            "target_energy":  0.88,
            "likes_acoustic": False,
            "strict_genre":   False,
            "strict_mood":    False,
        },
        "Chill Lofi": {
            "favorite_genre": "lofi",
            "favorite_mood":  "chill",
            "target_energy":  0.38,
            "likes_acoustic": True,
            "strict_genre":   False,
            "strict_mood":    False,
        },
        "Deep Intense Rock": {
            "favorite_genre": "rock",
            "favorite_mood":  "intense",
            "target_energy":  0.95,
            "likes_acoustic": False,
            "strict_genre":   False,
            "strict_mood":    False,
        },
    }

    for label, user_prefs in profiles.items():
        print_recommendations(label, user_prefs, songs, k=3)


if __name__ == "__main__":
    main()
