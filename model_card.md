# Model Card: Music Recommender Simulation

---

## 1. Model Name

**MoodMatch 1.0**

---

## 2. Intended Use

MoodMatch suggests songs based on how you're feeling and what energy you want right now. It's a classroom project — not a replacement for Spotify. You shouldn't use it to make real decisions about what music people will enjoy.

---

## 3. How the Model Works

Every song gets a score based on three things: mood, energy, and genre. Mood match earns 2 points, a close energy level earns up to 4 points, and genre match earns half a point. The song with the highest total gets recommended first.

Things like tempo, danceability, and acousticness are in the data but the scorer doesn't use them yet.

---

## 4. Data

The catalog has 18 songs across 15 genres and 14 moods. Most genres and moods only appear once. Lofi has three songs, pop has two, everything else has one.

Most songs lean toward low energy — 10 of 18 are below 0.55. There's almost nothing in the 0.55–0.75 range, so users who want that energy level don't get great results.

No songs were added or removed from the starter dataset.

---

## 5. Strengths

When your preferences match the catalog well — like chill lofi at low energy or intense rock at high energy — the right songs show up in the right order. The scoring is also easy to explain. You can look at any result and see exactly why it ranked where it did.

---

## 6. Limitations and Bias

The biggest issue is mood label mismatch. The scorer only gives points for an exact word match, so "relaxed" and "chill" score zero for each other even though they mean basically the same thing.

During testing, a user who wanted ambient and relaxed music got a jazz song as their top result — just because "relaxed" didn't match "chill" on the ambient song. That's the wrong answer for the wrong reason.

On top of that, 11 of the 14 moods only have one song each. So if your mood doesn't match exactly, you lose 2 points with nowhere to recover them, and the rest of your list fills up with random energy-sorted songs.

---

## 7. Evaluation

We tested 11 profiles total — 3 normal ones and 8 designed to break things.

The normal profiles (happy pop, chill lofi, intense rock) all returned sensible results. The adversarial ones showed some real problems:

- A blues/sad user with high energy got the right song at #1, then rock, EDM, and metal for the rest.
- A user with both strict filters on (classical AND angry) got a silent empty list — no songs, no explanation.
- The ambient/relaxed user got jazz first. The actual ambient song came second, because "relaxed" ≠ "chill."
- A folk fan who said they dislike acoustic music got the most acoustic song in the catalog recommended first.

**Why does "Gym Hero" keep showing up for happy pop users?**
It's a pop song with high energy, so it scores well on genre and energy. But its mood is "intense," not "happy." The system doesn't treat those as opposites — it just gives zero mood points and moves on. Strong energy + right genre floats it to #3 even though it sounds like a workout song, not a birthday party song.

---

## 8. Future Work

1. **Partial mood credit** — group similar moods like chill, relaxed, and peaceful together so close matches still earn some points instead of zero.
2. **Use the acoustic preference** — the field is already collected from the user and every song already has an acousticness score. Just connect them in the scorer.
3. **More songs** — most genres and moods have only one song. Adding two or three per mood would fix more problems than any weight change would.

---

## 9. Personal Reflection

**Biggest learning moment**
A one-word label difference completely changed who got a good recommendation. "Relaxed" vs "chill" — same feeling, different string, totally different result. That made it clear that a recommender isn't just a math problem. It's also a language problem and a data problem.

**How AI tools helped — and when I had to check**
AI helped me come up with adversarial test cases I wouldn't have thought of, like combining sad mood with high energy to see if the system would break. It also pointed out the energy gap in the catalog. But I had to verify every predicted score by actually running the code — a few were off because the tool was estimating from memory, not calculating. Always confirm against real output.

**What surprised me about simple algorithms**
The three normal profiles felt genuinely smart, even though the whole system is just three numbers added together. That was surprising. The "intelligence" came from the data being well-structured, not from anything clever in the algorithm. Real apps like Spotify are probably the same idea, just with a much bigger catalog.

**What I'd try next**
Group similar moods for partial credit, actually plug in the acoustic preference, and add more songs before touching anything else. Better data fixes more than better math does.
