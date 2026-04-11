from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

def _score_song(song: Song, user: UserProfile) -> float:
    """
    Points-based content score. Higher is better; max possible = 5.0.

    Scoring recipe:
        +2.0      mood match     (emotional fit — primary signal)
        +0.0..2.0 energy proximity = (1 - |song.energy - user.target_energy|) × 2
                  (scaled to 0..2; perfect energy match = 2.0)
        +1.0      genre match    (structural fit — secondary signal)

    Max score: 5.0
    """
    score = 0.0

    if song.mood == user.favorite_mood:
        score += 2.0

    energy_similarity = (1.0 - abs(song.energy - user.target_energy)) * 2.0
    score += energy_similarity

    if song.genre == user.favorite_genre:
        score += 1.0

    return round(score, 4)


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the catalog of songs for repeated recommendation queries."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked by score for the given user profile."""
        scored = sorted(self.songs, key=lambda s: _score_song(s, user), reverse=True)
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Build a human-readable explanation of why a song was recommended."""
        reasons = []

        if song.mood == user.favorite_mood:
            reasons.append(f"mood matches your preference for '{user.favorite_mood}'")
        if song.genre == user.favorite_genre:
            reasons.append(f"genre matches '{user.favorite_genre}'")

        energy_gap = abs(song.energy - user.target_energy)
        if energy_gap <= 0.15:
            reasons.append(f"energy ({song.energy}) is close to your target ({user.target_energy})")

        if not reasons:
            reasons.append("partial match across energy and acousticness")

        score = _score_song(song, user)
        return f"Score {score:.2f} — " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    int(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Algorithm Recipe — scores one song against user preferences.
    Returns (score, reasons) so callers can show why a song was recommended.

    Points awarded:
        +2.0      mood match     — primary signal (emotional fit)
        +0.0..2.0 energy proximity = (1 - |song.energy - target_energy|) × 2
        +1.0      genre match    — secondary signal (structural fit)

    Max score: 5.0
    """
    score = 0.0
    reasons = []

    # Mood match: +2.0
    if song.get("mood") == user_prefs.get("favorite_mood"):
        score += 2.0
        reasons.append(f"mood match: '{user_prefs.get('favorite_mood')}' (+2.0)")

    # Energy proximity: +0.0 to +2.0
    energy_similarity = (1.0 - abs(song.get("energy", 0.5) - user_prefs.get("target_energy", 0.5))) * 2.0
    score += energy_similarity
    reasons.append(f"energy proximity: {energy_similarity:.2f} (+{energy_similarity:.2f})")

    # Genre match: +1.0
    if song.get("genre") == user_prefs.get("favorite_genre"):
        score += 1.0
        reasons.append(f"genre match: '{user_prefs.get('favorite_genre')}' (+1.0)")

    return round(score, 4), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Expected return format: (song_dict, score, explanation)

    Strict filters (optional keys in user_prefs):
        strict_genre: True  → only score songs whose genre matches exactly
        strict_mood:  True  → only score songs whose mood matches exactly
    """
    strict_genre = user_prefs.get("strict_genre", False)
    strict_mood  = user_prefs.get("strict_mood",  False)

    def passes_filters(song: Dict) -> bool:
        """Return False if the song fails any active strict filter, True otherwise."""
        if strict_genre and song.get("genre") != user_prefs.get("favorite_genre"):
            return False
        if strict_mood and song.get("mood") != user_prefs.get("favorite_mood"):
            return False
        return True

    results = [
        (song, score, f"Score {score:.2f} — " + "; ".join(reasons))
        for song in songs
        if passes_filters(song)
        for score, reasons in [score_song(user_prefs, song)]
    ]

    return sorted(results, key=lambda x: x[1], reverse=True)[:k]
