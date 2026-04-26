"""
RAG-powered music recommender — works with or without the Groq API.

Flow (same in both modes):
  1. Extract structured preferences from the user's plain-English input
  2. Retrieve matching songs from the catalog (the RAG retrieval step)
  3. Generate a recommendation narrative grounded in those specific songs

Extraction & generation strategy:
  • GROQ_API_KEY set   → Groq Llama (function-calling extraction + LLM narrative)
  • No API key         → keyword-based NLP extraction + template narrative
                        (no extra dependencies beyond the standard library)
"""

import os
import json
import logging
import sys
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from recommender import score_song, recommend_songs

# ---------------------------------------------------------------------------
# Logging — writes to recommender.log AND stderr
# ---------------------------------------------------------------------------
log_path = os.path.join(os.path.dirname(__file__), "..", "recommender.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog vocabulary (used as guardrails for both paths)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Keyword-based NLP extractor (no API required)
# ---------------------------------------------------------------------------

# Maps words/phrases the user might say → catalog mood labels
_MOOD_KEYWORDS: dict[str, str] = {
    # chill / relaxed
    "chill": "chill", "calm": "chill", "low-key": "chill", "lowkey": "chill",
    "mellow": "chill", "easy": "chill", "laid-back": "chill", "laid back": "chill",
    "unwind": "chill", "wind down": "chill", "cozy": "chill", "cosy": "chill",
    "relax": "relaxed", "relaxed": "relaxed", "soothing": "relaxed", "gentle": "relaxed",
    # happy / upbeat
    "happy": "happy", "upbeat": "happy", "cheerful": "happy", "joyful": "happy",
    "fun": "happy", "bright": "happy", "positive": "happy", "good vibes": "happy",
    # focused / studying
    "focus": "focused", "focused": "focused", "study": "focused", "studying": "focused",
    "concentrate": "focused", "work": "focused", "productive": "focused", "coding": "focused",
    # energetic / intense
    "energetic": "energetic", "hype": "energetic", "pump": "energetic",
    "intense": "intense", "aggressive": "intense", "hard": "intense", "heavy": "intense",
    "workout": "energetic", "gym": "energetic", "run": "energetic", "running": "energetic",
    # sad / melancholic
    "sad": "sad", "melancholic": "melancholic", "melancholy": "melancholic",
    "blue": "sad", "blues": "sad", "emotional": "melancholic", "lonely": "melancholic",
    "heartbreak": "sad", "cry": "sad", "crying": "sad", "grief": "sad",
    # peaceful / calm
    "peaceful": "peaceful", "serene": "peaceful", "tranquil": "peaceful", "sleep": "peaceful",
    "meditat": "peaceful", "ambient": "peaceful", "nature": "peaceful",
    # moody
    "moody": "moody", "dark": "moody", "atmospheric": "moody", "late night": "moody",
    "night drive": "moody", "rainy": "moody", "rain": "moody",
    # romantic
    "romantic": "romantic", "love": "romantic", "date": "romantic",
    # nostalgic
    "nostalgic": "nostalgic", "nostalgia": "nostalgic", "throwback": "nostalgic",
    "childhood": "nostalgic", "memory": "nostalgic", "memories": "nostalgic",
    # confident
    "confident": "confident", "motivation": "confident", "motivated": "confident",
    "empowered": "confident", "boss": "confident",
    # angry
    "angry": "angry", "anger": "angry", "frustrat": "angry", "mad": "angry", "rage": "angry",
    # happy / quirky / playful
    "quirky": "happy", "playful": "happy", "whimsical": "happy", "silly": "happy",
    "funky": "happy", "groovy": "happy", "bouncy": "happy",
}

# Maps words/phrases → catalog genre labels
_GENRE_KEYWORDS: dict[str, str] = {
    "lofi": "lofi", "lo-fi": "lofi", "lo fi": "lofi",
    "pop": "pop",
    "rock": "rock",
    "metal": "metal",
    "hip hop": "hip-hop", "hip-hop": "hip-hop", "hiphop": "hip-hop", "rap": "hip-hop",
    "jazz": "jazz",
    "classical": "classical", "orchestra": "classical", "piano": "classical",
    "ambient": "ambient",
    "edm": "edm", "electronic": "edm", "dance": "edm", "house": "edm", "techno": "edm",
    "folk": "folk", "acoustic": "folk",
    "country": "country",
    "r&b": "r&b", "rnb": "r&b", "soul": "r&b",
    "blues": "blues",
    "synthwave": "synthwave", "synth": "synthwave", "retro": "synthwave",
    "indie": "indie pop", "indie pop": "indie pop",
}

# Maps activity/context words → approximate energy (0.0–1.0)
_ENERGY_HINTS: list[tuple[str, float]] = [
    # very low energy
    ("sleep", 0.15), ("sleeping", 0.15), ("meditation", 0.15), ("meditate", 0.15),
    ("nap", 0.20), ("quiet", 0.20), ("silent", 0.20),
    # low energy
    ("study", 0.30), ("studying", 0.30), ("reading", 0.30), ("background", 0.35),
    ("coffee", 0.35), ("rain", 0.30), ("rainy", 0.30), ("calm", 0.30), ("relax", 0.35),
    ("chill", 0.38), ("unwind", 0.35), ("morning", 0.40),
    # medium energy
    ("focus", 0.45), ("work", 0.45), ("coding", 0.45), ("commute", 0.55),
    ("walking", 0.55), ("drive", 0.60), ("driving", 0.60),
    # high energy
    ("upbeat", 0.75), ("happy", 0.72), ("fun", 0.72), ("party", 0.80),
    ("dance", 0.82), ("hype", 0.85),
    # very high energy
    ("workout", 0.90), ("gym", 0.92), ("run", 0.90), ("running", 0.90),
    ("pump", 0.90), ("intense", 0.93), ("hard", 0.90), ("aggressive", 0.93),
]

_DEFAULT_GENRE = "pop"
_DEFAULT_MOOD = "chill"
_DEFAULT_ENERGY = 0.5


def _match(keyword: str, text: str) -> bool:
    """
    True if keyword appears as a whole word or phrase in text.
    Uses word-boundary regex to prevent 'work' matching inside 'working'.
    Multi-word phrases like 'late night' are matched as a unit.
    """
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


def _prefix_match_first(tokens: list[str], keyword_dict: dict) -> Optional[str]:
    """Return the first dict value whose key starts with any token (min 3 chars)."""
    for tok in tokens:
        if len(tok) < 3:
            continue
        for kw, label in keyword_dict.items():
            if kw.startswith(tok):
                return label
    return None


def _keyword_extract(text: str) -> dict:
    """
    Rule-based preference extractor. No external dependencies.
    Scans the lowercased input for genre, mood, and energy hints.
    Uses whole-word matching so 'work' does not fire inside 'working'.
    Falls back to prefix matching for partial/misspelled words (e.g. 'quir' → 'quirky').
    """
    lower = text.lower()

    # Genre — first whole-word match wins
    genre = _DEFAULT_GENRE
    genre_matched = False
    for kw, label in _GENRE_KEYWORDS.items():
        if _match(kw, lower):
            genre = label
            genre_matched = True
            break

    # Mood — first whole-word match wins
    mood = _DEFAULT_MOOD
    mood_matched = False
    for kw, label in _MOOD_KEYWORDS.items():
        if _match(kw, lower):
            mood = label
            mood_matched = True
            break

    # Prefix fallback: only when no exact match was found at all
    # (guards against partial/misspelled words like 'quir' → 'quirky')
    if not genre_matched or not mood_matched:
        tokens = re.findall(r"\w+", lower)
        if not genre_matched:
            label = _prefix_match_first(tokens, _GENRE_KEYWORDS)
            if label:
                genre = label
        if not mood_matched:
            label = _prefix_match_first(tokens, _MOOD_KEYWORDS)
            if label:
                mood = label

    # Energy — average all hints that fire, fall back to default
    matched_energies = [e for kw, e in _ENERGY_HINTS if _match(kw, lower)]
    energy = round(sum(matched_energies) / len(matched_energies), 2) if matched_energies else _DEFAULT_ENERGY

    # Acoustic preference
    likes_acoustic = any(_match(w, lower) for w in ("acoustic", "unplugged", "organic", "guitar"))

    reasoning = (
        f"Keyword scan: genre='{genre}' (from genre words), "
        f"mood='{mood}' (from mood/activity words), "
        f"energy={energy:.2f} (averaged from {len(matched_energies)} activity hints)"
    )

    logger.info("Keyword extraction | genre=%s mood=%s energy=%.2f", genre, mood, energy)
    return {
        "favorite_genre": genre,
        "favorite_mood": mood,
        "target_energy": energy,
        "likes_acoustic": likes_acoustic,
        "reasoning": reasoning,
    }


def _template_response(user_input: str, prefs: dict, candidates: list) -> str:
    """
    Build a readable recommendation paragraph from retrieved songs without an LLM.
    The response references actual retrieved songs — it changes if retrieval changes.
    """
    if not candidates:
        return "Sorry, nothing in the catalog matched your request. Try different keywords."

    top_song, top_score, _ = candidates[0]
    mood = prefs["favorite_mood"]
    genre = prefs["favorite_genre"]
    energy = prefs["target_energy"]

    # Describe how well the top match fits
    if top_score >= 4.0:
        fit = "a great match"
    elif top_score >= 3.0:
        fit = "a solid match"
    else:
        fit = "the closest we have"

    lines = [
        f'Based on your request for {mood}, {genre}-leaning music at energy {energy:.1f}, '
        f'here\'s what fits best from the catalog.',
        "",
        f'Your top pick is "{top_song["title"]}" by {top_song["artist"]} — '
        f'{fit} ({top_song["genre"]}, {top_song["mood"]}, energy {top_song["energy"]:.2f}).',
    ]

    if len(candidates) >= 2:
        s2, sc2, _ = candidates[1]
        lines.append(
            f'Also worth trying: "{s2["title"]}" by {s2["artist"]} '
            f'({s2["genre"]}, {s2["mood"]}).'
        )
    if len(candidates) >= 3:
        s3, sc3, _ = candidates[2]
        lines.append(
            f'"{s3["title"]}" by {s3["artist"]} rounds out the list '
            f'if you want more variety.'
        )

    # Honest caveat when the top score is low
    if top_score < 2.5:
        lines.append(
            "\nNote: the catalog doesn't have a perfect match for this combination — "
            "these are the best available options."
        )

    lines.append(
        f'\nOverall vibe: {mood} energy with a {genre} lean, sitting around {energy:.1f}/1.0 intensity.'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------
def confidence_score(candidates: list) -> float:
    """
    Return a 0.0–1.0 confidence score for a retrieval result.

    Combines two signals:
      - Base score:  top_score / 5.0  (how well the best song matched)
      - Gap bonus:   up to +0.15 when there is a clear winner (gap between
                     #1 and #2 is large — the system is not just guessing)

    Thresholds:
      >= 0.80  high confidence  (mood + genre + energy all aligned)
      0.60–0.79  moderate       (1–2 signals matched)
      < 0.60   low              (energy-only matches, catalog gap)
    """
    if not candidates:
        return 0.0
    top_score = candidates[0][1]
    base = top_score / 5.0
    if len(candidates) >= 2:
        gap = top_score - candidates[1][1]
        gap_bonus = min(gap, 1.0) * 0.15
        return round(min(base + gap_bonus, 1.0), 2)
    return round(base, 2)


# ---------------------------------------------------------------------------
# Groq API path (used only when GROQ_API_KEY is set)
# ---------------------------------------------------------------------------
_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_music_preferences",
        "description": "Extract structured music preferences from the user's natural language description.",
        "parameters": {
            "type": "object",
            "properties": {
                "favorite_genre": {
                    "type": "string",
                    "enum": VALID_GENRES,
                    "description": f"Best-matching genre from: {', '.join(VALID_GENRES)}",
                },
                "favorite_mood": {
                    "type": "string",
                    "enum": VALID_MOODS,
                    "description": f"Best-matching mood from: {', '.join(VALID_MOODS)}",
                },
                "target_energy": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Energy 0.0 (very calm) to 1.0 (very intense). Infer from context.",
                },
                "likes_acoustic": {
                    "type": "boolean",
                    "description": "True if user prefers acoustic/unplugged sounds.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining your mapping choices.",
                },
            },
            "required": ["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic", "reasoning"],
        },
    },
}

_SYSTEM_PROMPT = """\
You are a warm, knowledgeable music recommendation assistant.

You will receive the user's original request, extracted preferences, and a ranked
list of songs ALREADY RETRIEVED from the catalog.

Write a 3–5 sentence conversational recommendation that:
  - Names specific songs and artists from the retrieved list (exact titles only)
  - Explains WHY each top pick fits the user's mood or activity
  - Honestly notes imperfect matches ("closest we have is…")
  - Ends with one sentence describing the overall vibe

Rules: only reference songs from the retrieved list. Do not invent songs.
"""


def _groq_extract(user_input: str, client) -> Optional[dict]:
    """Use Groq function calling to extract structured preferences."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=512,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_music_preferences"}},
            messages=[{"role": "user", "content": f"Extract music preferences: {user_input}"}],
        )
        tool_call = response.choices[0].message.tool_calls
        if tool_call:
            prefs = json.loads(tool_call[0].function.arguments)
            prefs["target_energy"] = max(0.0, min(1.0, float(prefs["target_energy"])))
            logger.info("Groq extracted prefs | %s", prefs)
            return prefs
    except Exception as exc:
        logger.error("Groq extraction failed | %s", exc)
    return None


def _groq_generate(user_input: str, prefs: dict, candidates: list, client) -> str:
    """Use Groq to write a grounded narrative from retrieved songs."""
    context = "\n".join(
        f'{i}. "{s["title"]}" by {s["artist"]} '
        f'[{s["genre"]}, {s["mood"]}, energy={s["energy"]:.2f}] score={sc:.2f}/5.00'
        for i, (s, sc, _) in enumerate(candidates, 1)
    )
    user_message = (
        f'User request: "{user_input}"\n'
        f"Interpreted: genre={prefs['favorite_genre']}, mood={prefs['favorite_mood']}, "
        f"energy={prefs['target_energy']:.1f}\n"
        f"Reasoning: {prefs.get('reasoning', '')}\n\n"
        f"Retrieved songs:\n{context}\n\nWrite your recommendation."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content
        logger.info("Groq narrative generated | %d chars", len(text))
        return text
    except Exception as exc:
        logger.error("Groq generation failed | %s", exc)
        return _template_response(user_input, prefs, candidates)


# ---------------------------------------------------------------------------
# Shared retrieval step (same for both paths)
# ---------------------------------------------------------------------------
def retrieve_candidates(user_prefs: dict, songs: list, k: int = 5) -> list:
    """Score all catalog songs and return the top-k matches."""
    logger.info(
        "Retrieving | genre=%s mood=%s energy=%.2f",
        user_prefs["favorite_genre"], user_prefs["favorite_mood"], user_prefs["target_energy"],
    )
    candidates = recommend_songs(user_prefs, songs, k=k)
    logger.info("Retrieved %d candidates", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------
def run_ai_mode(songs: list) -> None:
    """Entry point for the RAG-powered recommender (works with or without API key)."""
    api_key = os.environ.get("GROQ_API_KEY")

    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            mode_label = "Groq Llama-powered"
            logger.info("AI mode: Groq API")
        except ImportError:
            client = None
            mode_label = "keyword-based (groq package not installed)"
            logger.warning("groq package missing — falling back to keyword mode")
    else:
        client = None
        mode_label = "keyword-based (no API key)"
        logger.info("AI mode: keyword-based fallback")

    print("\n" + "=" * 62)
    print(f"  AI Music Recommender  [{mode_label}]")
    print("  Describe what you want in plain English.")
    print("  Type 'quit' to exit.")
    print("=" * 62)

    while True:
        print()
        try:
            user_input = input("What kind of music are you looking for?  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q", ""):
            print("Goodbye!")
            break

        # ── Step 1: Extract preferences ──────────────────────────────────
        print("\n  [Interpreting your request...]")
        if client is not None:
            prefs = _groq_extract(user_input, client)
            if prefs is None:
                logger.warning("Groq extraction returned None — falling back to keyword")
                prefs = _keyword_extract(user_input)
        else:
            prefs = _keyword_extract(user_input)

        print(
            f"  -> Genre: {prefs['favorite_genre']}  |  "
            f"Mood: {prefs['favorite_mood']}  |  "
            f"Energy: {prefs['target_energy']:.1f}"
        )

        # ── Step 2: Retrieve matching songs ───────────────────────────────
        print("  [Searching catalog...]")
        candidates = retrieve_candidates(prefs, songs, k=5)

        if not candidates:
            print("  [No songs matched — the catalog may not cover this niche.]")
            logger.warning("Empty candidates | prefs=%s", prefs)
            continue

        # ── Step 3: Generate grounded narrative ──────────────────────────
        print("  [Generating recommendation...]\n")
        if client is not None:
            narrative = _groq_generate(user_input, prefs, candidates, client)
        else:
            narrative = _template_response(user_input, prefs, candidates)

        print("-" * 62)
        print(narrative)
        print("-" * 62)

        # Structured rankings alongside narrative
        print("\n  Full ranked list:")
        for rank, (song, score, _) in enumerate(candidates, 1):
            _, reasons = score_song(prefs, song)
            signals = [r for r in reasons if not r.startswith("energy proximity")]
            tag = " | ".join(signals) if signals else "energy proximity only"
            print(f"    #{rank}  {song['title']} by {song['artist']}  [{score:.2f}/5.00]")
            print(f"         {tag}")

        logger.info("Recommendation complete | query=%r | mode=%s", user_input, mode_label)
