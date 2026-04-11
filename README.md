# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world recommenders like Spotify and YouTube primarily use a hybrid approach — combining content-based filtering (analyzing the song itself) with collaborative filtering (learning from what millions of users listen to). My version focuses on **content-based filtering**: it compares each song's attributes directly against a user's stated preferences, without relying on any other users' behavior.

**Song features used:**
- `genre` — the musical category (e.g. lofi, pop, rock)
- `mood` — the emotional character (e.g. chill, happy, intense)
- `energy` — a 0–1 scale of intensity
- `acousticness` — a 0–1 scale of how acoustic vs. produced the song sounds

**UserProfile stores:**
- `favorite_genre` — the genre the user prefers
- `favorite_mood` — the mood the user is looking for
- `target_energy` — the energy level the user wants (0–1)
- `likes_acoustic` — whether the user prefers acoustic-sounding songs

**Algorithm Recipe:** Each song is scored on a 0–5 point scale using three independent signals, evaluated in priority order:

| Signal | Points | How it is calculated |
|---|---|---|
| Mood match | +2.0 (flat) | Exact string match between `song.mood` and `user.favorite_mood` |
| Energy proximity | +0.0 → +2.0 | `(1 − |song.energy − user.target_energy|) × 2` — continuous, decays linearly from 2.0 (perfect) to 0.0 (opposite) |
| Genre match | +1.0 (flat) | Exact string match between `song.genre` and `user.favorite_genre` |
| **Max total** | **5.0** | |

Mood and energy together account for 4 of the 5 possible points, making emotional fit the primary driver of every recommendation. Genre acts as a tiebreaker rather than the lead signal.

**Ranking:** All songs in the catalog are scored independently, then sorted highest to lowest. The top `k` songs (default 5) are returned as recommendations.

**Potential biases to watch for:**

- **Mood-label dominance** — because a mood match is worth +2.0 flat, a song with the exact right mood label will almost always outrank a song that matches genre, energy, *and* acousticness but uses a slightly different mood word (e.g. `"relaxed"` vs `"chill"`). The system is sensitive to label vocabulary, not actual feeling.
- **Energy pulls toward the middle** — the linear proximity formula scores a song at energy 0.5 higher than one at 0.95 when the user's target is 0.7, even though 0.95 might subjectively feel closer. Songs with moderate energy tend to accumulate more energy points on average than extreme-energy songs.
- **Genre is a weak tiebreaker** — at only +1.0, genre can be completely overridden by a strong energy match in the wrong genre. A user who says "I want rock" could still receive a lofi result if its energy and mood scores are high enough.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

**Weight shift — doubling energy, halving genre:**
The energy multiplier was changed from ×2 to ×4 (max energy score went from +2.0 to +4.0) and the genre bonus was cut from +1.0 to +0.5. Running the three standard profiles showed identical rankings but higher raw scores. The most notable effect was that songs with a perfect energy match (like Drop Zone at energy 0.95 for a user targeting 0.95) jumped from 2.0 to 4.0 points, making pure energy-match songs much more competitive even with zero mood or genre alignment. The overall conclusion was that the original weights were more balanced.

**Adversarial profiles — 8 edge cases:**
Eight profiles were designed to expose specific weaknesses: conflicting signals (sad mood + high energy), strict filters with no matching songs, mood vocabulary mismatches (relaxed vs. chill), energy floor targets (0.0), ignored acoustic preferences, tie scenarios, and cases where energy completely overrides genre. The strict filter profile returned a silent empty list. The ambient/relaxed profile returned jazz as its top result. The folk/anti-acoustic profile returned the most acoustic song in the catalog first.

---

### Standard Profile Outputs

```
==================================================
  High-Energy Pop
  Genre: pop  |  Mood: happy  |  Energy: 0.88
==================================================
  #1  Sunrise City by Neon Echo          Score: 4.88 / 5.00
        • mood match: 'happy' (+2.0)
        • energy proximity: 1.88
        • genre match: 'pop' (+1.0)

  #2  Rooftop Lights by Indigo Parade    Score: 3.76 / 5.00
        • mood match: 'happy' (+2.0)
        • energy proximity: 1.76

  #3  Gym Hero by Max Pulse              Score: 2.90 / 5.00
        • energy proximity: 1.90
        • genre match: 'pop' (+1.0)
        (note: mood is 'intense', not 'happy' — no mood points)

==================================================
  Chill Lofi
  Genre: lofi  |  Mood: chill  |  Energy: 0.38
==================================================
  #1  Library Rain by Paper Lanterns     Score: 4.94 / 5.00
        • mood match: 'chill' (+2.0)
        • energy proximity: 1.94
        • genre match: 'lofi' (+1.0)

  #2  Midnight Coding by LoRoom          Score: 4.92 / 5.00
        • mood match: 'chill' (+2.0)
        • energy proximity: 1.92
        • genre match: 'lofi' (+1.0)

  #3  Spacewalk Thoughts by Orbit Bloom  Score: 3.80 / 5.00
        • mood match: 'chill' (+2.0)
        • energy proximity: 1.80
        (note: ambient genre, not lofi — no genre points)

==================================================
  Deep Intense Rock
  Genre: rock  |  Mood: intense  |  Energy: 0.95
==================================================
  #1  Storm Runner by Voltline           Score: 4.92 / 5.00
        • mood match: 'intense' (+2.0)
        • energy proximity: 1.92
        • genre match: 'rock' (+1.0)

  #2  Gym Hero by Max Pulse              Score: 3.96 / 5.00
        • mood match: 'intense' (+2.0)
        • energy proximity: 1.96
        (note: pop genre, not rock — no genre points)

  #3  Drop Zone by Bassline Cult         Score: 2.00 / 5.00
        • energy proximity: 2.00 (perfect energy match)
        (note: edm genre, energetic mood — no mood or genre points)
```

---

### Adversarial Profile Outputs

```
============================================================
  Profile 1 — Conflicting Signals (blues/sad, energy=0.9)
  Genre: blues  |  Mood: sad  |  Energy: 0.9
============================================================
  #1  Empty Glass by Roy Delmar     Score: 3.86  [MOOD + GENRE MATCH]
  #2  Storm Runner by Voltline      Score: 1.98  (energy only — rock)
  #3  Gym Hero by Max Pulse         Score: 1.94  (energy only — pop)
  #4  Drop Zone by Bassline Cult    Score: 1.90  (energy only — edm)
  #5  Shatter Everything            Score: 1.86  (energy only — metal)

  → Right song at #1, then pure energy filler from wrong genres.

============================================================
  Profile 2 — Strict Filter Blackhole (classical + angry, both strict)
============================================================
  *** EMPTY RESULT — no songs survived strict filters ***

  → Silent empty list. No explanation given to the user.

============================================================
  Profile 3 — Mood Label Trap (ambient/relaxed, energy=0.3)
  Genre: ambient  |  Mood: relaxed  |  Energy: 0.3
============================================================
  #1  Coffee Shop Stories (jazz, relaxed)    Score: 3.86  [MOOD MATCH]
  #2  Spacewalk Thoughts  (ambient, chill)   Score: 2.96  [GENRE MATCH]
  #3  Autumn Letters      (folk, melancholic) Score: 1.98
  #4  Empty Glass         (blues, sad)        Score: 1.94
  #5  Library Rain        (lofi, chill)       Score: 1.90

  → Jazz beats ambient because "relaxed" ≠ "chill" as strings.

============================================================
  Profile 4 — Energy Floor (classical/peaceful, target=0.0)
  Genre: classical  |  Mood: peaceful  |  Energy: 0.0
============================================================
  #1  Morning Prelude (classical, peaceful, 0.22)  Score: 4.56  [MOOD + GENRE]
  #2  Spacewalk Thoughts (ambient, chill, 0.28)    Score: 1.44
  #3  Autumn Letters     (folk, melancholic, 0.31) Score: 1.38
  #4  Empty Glass        (blues, sad, 0.33)        Score: 1.34
  #5  Library Rain       (lofi, chill, 0.35)       Score: 1.30

  → Right song at #1 but max possible is 4.56, not 5.0 — no song
    has energy exactly 0.0 so a perfect score is unreachable.

============================================================
  Profile 5 — Acousticness Ghost (folk/melancholic, likes_acoustic=False)
  Genre: folk  |  Mood: melancholic  |  Energy: 0.3
============================================================
  #1  Autumn Letters (folk, melancholic, acousticness=0.91)  Score: 4.98
  #2  Spacewalk Thoughts (ambient, acousticness=0.92)        Score: 1.96
  #3  Empty Glass        (blues,  acousticness=0.83)         Score: 1.94
  #4  Library Rain       (lofi,   acousticness=0.86)         Score: 1.90
  #5  Coffee Shop Stories (jazz,  acousticness=0.89)         Score: 1.86

  → User said no acoustic music. Top 5 are all highly acoustic.
    likes_acoustic field is never read by the scorer.

============================================================
  Profile 6 — Tie Ambush (lofi/focused, energy=0.60)
  Genre: lofi  |  Mood: focused  |  Energy: 0.6
============================================================
  #1  Focus Flow      (lofi, focused, 0.40)  Score: 4.60  [MOOD + GENRE]
  #2  Midnight Coding (lofi, chill,   0.42)  Score: 2.64  [GENRE]
  #3  Library Rain    (lofi, chill,   0.35)  Score: 2.50  [GENRE]
  #4  Slow Burn       (r&b, romantic, 0.55)  Score: 1.90
  #5  Dirt Road Memory (country, nostalgic)  Score: 1.76

  → Country appears at #5 with no mood or genre match — pure
    energy proximity floated it above other songs.

============================================================
  Profile 7 — Energy Override (rock/sad, target=0.95)
  Genre: rock  |  Mood: sad  |  Energy: 0.95
============================================================
  #1  Storm Runner  (rock, intense, 0.91)   Score: 2.92  [GENRE]
  #2  Empty Glass   (blues, sad,    0.33)   Score: 2.76  [MOOD]
  #3  Drop Zone     (edm, energetic, 0.95)  Score: 2.00
  #4  Gym Hero      (pop, intense,   0.93)  Score: 1.96
  #5  Shatter Everything (metal, angry)     Score: 1.96

  → No single song satisfies both rock AND sad. Genre and mood
    pull the list in opposite directions.

============================================================
  Profile 8 — Middle Energy Attractor (hip-hop/angry, target=0.5)
  Genre: hip-hop  |  Mood: angry  |  Energy: 0.5
============================================================
  #1  Shatter Everything (metal, angry,  0.97)  Score: 3.06  [MOOD]
  #2  Crown Up           (hip-hop, confident)   Score: 2.44  [GENRE]
  #3  Dirt Road Memory   (country, nostalgic)   Score: 1.96
  #4  Slow Burn          (r&b, romantic)        Score: 1.90
  #5  Midnight Coding    (lofi, chill)          Score: 1.84

  → Hip-hop user gets metal at #1, country at #3, lofi at #5.
    Middle energy target pulls in unrelated genres.
```

---

## Limitations and Risks

- The catalog has only 18 songs. Most genres and moods have just one representative, so recommendations quickly run out of real matches and fall back to energy-sorted songs from unrelated genres.
- Mood matching is all-or-nothing. "Relaxed" and "chill" score zero for each other even though they describe the same feeling. This means the vocabulary used when tagging songs determines who gets good results, not actual musical similarity.
- The `likes_acoustic` user preference is collected but never used in scoring. A user who dislikes acoustic music will still receive highly acoustic songs with no penalty.
- There is an energy gap in the catalog between 0.55 and 0.75 with almost no songs. Users targeting that range are structurally under-served regardless of how the weights are set.
- Strict filters can produce a completely empty result with no explanation shown to the user.

See [model_card.md](model_card.md) for a deeper analysis.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

The most surprising thing about this project was how convincing the results felt for normal user profiles, even though the entire system is just three numbers added together. When the chill lofi user got calm, quiet songs at the top it genuinely felt like the system understood them — but it did not. It found songs whose labels and energy values happened to align numerically. That feeling of intelligence came from the data being well-structured, not from anything smart inside the algorithm.

The clearest lesson about bias was the ambient-gets-jazz failure. A user who asked for ambient and relaxed music received a jazz recommendation first because the word "relaxed" did not exactly match the word "chill" on the ambient song. Nothing in the code was wrong. The math was correct. The unfairness came from a labeling inconsistency in the data — and that kind of invisible bias is hard to catch without deliberately trying to break your own system.



