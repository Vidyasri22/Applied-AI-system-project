# Music Recommender — RAG-Powered AI Extension

A content-based music recommendation system extended with a natural language interface
and Retrieval-Augmented Generation (RAG) pipeline.

---

## Portfolio Artifact

- **GitHub:** https://github.com/Vidyasri22/Applied-AI-system-project
- **Final Video Link:** https://www.loom.com/share/9dadd79eb0ff4fcc84658064b33b74ae

### What this project says about me as an AI engineer

I build AI systems that are honest about their limits. This project could have been a
slick chatbot that always sounded confident — instead I designed it as a RAG pipeline
with a visible scoring layer, a keyword fallback so it runs without an API key, and an
evaluation harness that keeps one test deliberately failing to document a known bias.
I care more about a system a user can push back on than one that always sounds right,
and I treat retrieval quality, guardrails, and adversarial testing as first-class
concerns rather than polish added at the end.

---

## 1. Original Project

**Base project:** Music Recommender Simulation (CodePath Applied AI, Modules 1–3)

The original system was a rule-based content-based filtering recommender. Users selected
a genre, mood, and energy level from fixed dropdown lists, and the system scored every
song in an 18-song catalog using a weighted formula (mood +2.0, energy proximity up to
+2.0, genre +1.0 — max 5.0). It returned a ranked list of songs with per-song
explanations of why each was recommended. The project deliberately documented known
biases such as exact-string mood matching, an unused acousticness field, and catalog
imbalance (lofi had 3 songs; most genres had 1).

---

## 2. What This Extended Version Does — and Why It Matters

The extension adds a **RAG (Retrieval-Augmented Generation) pipeline** that lets users
describe what they want in plain English instead of filling out a form.

**Before (original):**
```
Available moods: angry, chill, confident, energetic, focused, happy...
Your favorite mood: chill
Target energy (0.0-1.0): 0.38
```

**After (AI mode):**
```
What kind of music are you looking for?
> I'm studying for finals and need something calm but focused
```

This matters for two reasons. First, real recommendation interfaces (Spotify, YouTube
Music) are moving toward conversational input — understanding what "studying for finals"
means in terms of genre, mood, and energy is a core NLP problem. Second, the extension
demonstrates a complete RAG architecture: the AI response is built from documents
retrieved from the catalog, not generated from memory. If the catalog changes, the
response changes. That grounding property is what separates RAG from a chatbot that
simply invents answers.

The system works with or without an API key. Without one, a keyword-based NLP extractor
handles preference extraction and a template generator handles the response — no
external dependencies required.

Two ways to use it:

- **Streamlit UI** (`streamlit run app.py`) — a clean visual interface that shows the
  extracted preferences as metric cards, the narrative in a grounded-output panel, and
  each ranked song as a scored card. Guardrails surface as a dedicated panel whenever
  one fires. Recommended for the walkthrough.
- **CLI** (`python src/main.py`) — a terminal-only version that runs the same pipeline
  and is useful for scripted / headless testing.

Both surfaces call the same extractor, retriever, generator, and guardrails — the UI
is a display layer, not a second pipeline.

---

## 3. Architecture Overview

```
USER INPUT
"I need calm music for studying"
        |
        v
+---------------------+
|  app.py (Streamlit) |   <-- visual UI (recommended)
|  OR src/main.py     |   <-- CLI entry / mode selector
+----------+----------+
         |
    .----|----.
    |         |
  Mode 1    Mode 2
  Classic    AI / RAG
    |         |
    |         v
    |  +-----------------------------+
    |  |   PREFERENCE EXTRACTOR      |
    |  |                             |
    |  |  No API key -> Keyword NLP  |  rule-based scan for
    |  |  API key    -> Groq Llama    |  genre, mood, energy
    |  |               (fn calling)  |  keywords in input
    |  +-------------+---------------+
    |                |
    |         GUARDRAIL: energy clamped to [0.0, 1.0]
    |         GUARDRAIL: API key missing -> auto-fallback
    |                |
    |                v
    |    { genre, mood, target_energy }
    |                |
    |                v
    |  +-----------------------------+     +------------+
    |  |        RETRIEVER            |<----|  songs.csv |
    |  |   recommender.py            |     | (18 songs) |
    |  |   score_song() x 18 songs  |     +------------+
    |  |   recommend_songs()         |
    |  +-------------+---------------+
    |                |
    |         GUARDRAIL: empty result -> warn user, skip generation
    |                |
    |                v
    |         top-5 candidates + match scores
    |                |
    |                v
    |  +-----------------------------+
    |  |     RESPONSE GENERATOR      |
    |  |                             |
    |  |  No API key -> Template     |  references actual
    |  |  API key    -> Groq Llama    |  retrieved songs
    |  |               (narrative)   |  (RAG: output changes
    |  +-------------+---------------+  if retrieval changes)
    |                |
    `------> OUTPUT <-'
             |  - AI narrative (grounded in retrieved songs)
             |  - Ranked list with per-song match reasons
             v
      recommender.log  (every step timestamped)


HUMAN & TESTING CHECKPOINTS
------------------------------------------------------------

  [1] pytest                [2] adversarial_test.py   [3] Human
      test_recommender.py       8 edge-case profiles       reads
      unit tests for            bias detection runs         output
      OOP interface             (empty results, label       and
      (runs before deploy)      traps, energy drift)        judges
            |                         |                quality
            '-------------------------'
                          |
                   pass / fail / bias flagged
```

**Component summary:**

| Component | File | Role |
|---|---|---|
| Streamlit UI | `app.py` | Visual interface — wraps the same pipeline as a web app |
| CLI entry | `src/main.py` | Terminal entry; picks Classic or AI mode at startup |
| Preference Extractor | `src/ai_recommender.py` | Keyword NLP scan or Groq Llama function calling |
| Retriever | `src/recommender.py` | Scores all 18 songs, returns top-k |
| Response Generator | `src/ai_recommender.py` | Template or Groq Llama narrative, grounded in retrieved songs |
| Catalog | `data/songs.csv` | 18 songs, 10 audio features each |
| Logger / Guardrails | `src/ai_recommender.py` | Energy clamping, fallback chain, empty-result check, log file |
| Unit tests | `tests/test_recommender.py` | pytest — OOP interface correctness |
| Bias tests | `src/adversarial_test.py` | 8 adversarial profiles — human-reviewed |

The key design insight: **all three steps share a strict boundary**. The extractor only
produces structured preferences. The retriever only reads the catalog and returns scored
songs. The generator only writes a narrative from what retrieval gave it — it cannot
invent songs. This is what makes the system RAG rather than a chatbot.

---

## 4. Setup Instructions

### Prerequisites

- Python 3.10 or higher
- A Groq API key (optional — the system runs without one)

### Installation

**Step 1** — Clone the repository and enter the project folder:

```bash
git clone <repo-url>
cd Applied-AI-system-project
```

**Step 2** — Create and activate a virtual environment:

```bash
python -m venv .venv

source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows Command Prompt
.venv\Scripts\Activate.ps1       # Windows PowerShell
```

**Step 3** — Install dependencies:

```bash
pip install -r requirements.txt
```

**Step 4 (AI mode only)** — Add your Groq API key to a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com/).
If you skip this step the system still runs — it uses keyword NLP instead of Groq Llama.

**Step 5** — Run the application. Two options:

**Option A — Streamlit UI (recommended):**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. You get a visual interface with quick-pick example
queries, extracted preferences shown as metric cards, the grounded narrative in a
highlighted panel, and each ranked song as a scored card. The sidebar shows whether
you're running in Groq or keyword-fallback mode, plus a list of active guardrails.

**Option B — CLI:**

```bash
cd src
python main.py
```

You will see:

```
Loaded 18 songs from catalog.

Select mode:
  1  Classic - run standard profiles
  2  AI mode - describe what you want in plain English (requires GROQ_API_KEY)

Enter 1 or 2:
```

### Running Tests

```bash
# From the project root
pytest
```

To run the adversarial bias-detection suite:

```bash
cd src
python adversarial_test.py
```

---

## 5. Sample Interactions

### Classic Mode — Example

**Input:** High-Energy Pop profile (genre=pop, mood=happy, energy=0.88)

**Output:**
```
==================================================
  High-Energy Pop
  Genre: pop  |  Mood: happy  |  Energy: 0.88
==================================================
  #1  Sunrise City by Neon Echo          Score: 4.88 / 5.00
        mood match: 'happy' (+2.0)
        energy proximity: 1.88
        genre match: 'pop' (+1.0)

  #2  Rooftop Lights by Indigo Parade    Score: 3.76 / 5.00
        mood match: 'happy' (+2.0)
        energy proximity: 1.76

  #3  Gym Hero by Max Pulse              Score: 2.90 / 5.00
        energy proximity: 1.90
        genre match: 'pop' (+1.0)
        (note: mood is 'intense', not 'happy' — no mood points)
```

---

### AI Mode — Example 1 (no API key)

**Input:**
```
What kind of music are you looking for?
> I'm studying for finals and need something calm but focused
```

**Output:**
```
  -> Genre: pop  |  Mood: chill  |  Energy: 0.3

  ------------------------------------------------------------
  Based on your request for chill, pop-leaning music at energy 0.3,
  here's what fits best from the catalog.

  Your top pick is "Library Rain" by Paper Lanterns — a solid match
  (lofi, chill, energy 0.35).
  Also worth trying: "Spacewalk Thoughts" by Orbit Bloom (ambient, chill).
  "Midnight Coding" by LoRoom rounds out the list if you want more variety.

  Overall vibe: chill energy with a pop lean, sitting around 0.3/1.0 intensity.
  ------------------------------------------------------------

  Full ranked list:
    #1  Library Rain by Paper Lanterns   [3.98/5.00]
         mood match: 'chill' (+2.0)
    #2  Spacewalk Thoughts by Orbit Bloom [3.88/5.00]
         mood match: 'chill' (+2.0)
    #3  Midnight Coding by LoRoom         [3.84/5.00]
         mood match: 'chill' (+2.0)
```

What happened: "studying" and "calm" together pulled energy to 0.34. "Focused" and
"calm" both matched the `chill` mood label. Even without a genre signal in the input,
the mood+energy combination surfaced the three chill catalog songs at the top.

---

### AI Mode — Example 2 (no API key)

**Input:**
```
What kind of music are you looking for?
> I need something moody for a late night drive
```

**Output:**
```
  -> Genre: pop  |  Mood: moody  |  Energy: 0.6

  ------------------------------------------------------------
  Based on your request for moody, pop-leaning music at energy 0.6,
  here's what fits best from the catalog.

  Your top pick is "Night Drive Loop" by Neon Echo — a solid match
  (synthwave, moody, energy 0.75).
  Also worth trying: "Sunrise City" by Neon Echo (pop, happy).
  "Gym Hero" by Max Pulse rounds out the list if you want more variety.
  ------------------------------------------------------------

  Full ranked list:
    #1  Night Drive Loop by Neon Echo   [3.70/5.00]
         mood match: 'moody' (+2.0)
    #2  Sunrise City by Neon Echo       [2.56/5.00]
         genre match: 'pop' (+1.0)
    #3  Gym Hero by Max Pulse           [2.34/5.00]
         genre match: 'pop' (+1.0)
```

What happened: "moody" matched the `moody` label directly. "Late night drive" and
"night" hints averaged to energy 0.60. The top result — a synthwave song literally
called "Night Drive Loop" — was correct entirely because its mood label matched.

---

### AI Mode — Example 3: Guardrail firing (no API key)

**Input:**
```
What kind of music are you looking for?
> something really aggressive and angry for the gym
```

**Output:**
```
  -> Genre: pop  |  Mood: angry  |  Energy: 0.91

  ------------------------------------------------------------
  Based on your request for angry, pop-leaning music at energy 0.9,
  here's what fits best from the catalog.

  Your top pick is "Shatter Everything" by Iron Hollow — a solid match
  (metal, angry, energy 0.97).
  Also worth trying: "Storm Runner" by Voltline (rock, intense).
  "Gym Hero" by Max Pulse rounds out the list if you want more variety.

  Note: the catalog doesn't have a perfect match for this combination —
  these are the best available options.
  ------------------------------------------------------------
```

What happened: "angry" matched the `angry` label. "Gym" and "aggressive" both pushed
energy above 0.9. The guardrail note fired because no single song in the catalog is
both `angry` mood and `pop` genre — the system honestly disclosed the gap rather than
silently returning an imperfect list.

---

### AI Mode — Example 4: Groq Llama-powered (with API key)

**Input:**
```
What kind of music are you looking for?
> I want something moody and atmospheric for a rainy evening
```

**Output:**
```
  [Interpreting your request...]
  -> Genre: pop  |  Mood: moody  |  Energy: 0.3

  [Searching catalog...]
  [Generating recommendation...]

  --------------------------------------------------------------
  For a rainy, moody evening, "Night Drive Loop" by Neon Echo is
  your best match — its dark synthwave atmosphere and brooding energy
  make it perfect for settling into a stormy night. If you want
  something that leans a little softer, "Spacewalk Thoughts" by
  Orbit Bloom offers a similar introspective quality with a more
  ambient texture. "Library Rain" by Paper Lanterns rounds things
  out — the name alone tells you it was made for exactly this mood.

  Overall vibe: atmospheric and low-key, with just enough texture
  to fill the room without demanding attention.
  --------------------------------------------------------------

  Full ranked list:
    #1  Night Drive Loop by Neon Echo   [3.70/5.00]
         mood match: 'moody' (+2.0)
    #2  Spacewalk Thoughts by Orbit Bloom [3.56/5.00]
         mood match: 'chill' (+2.0)
    #3  Library Rain by Paper Lanterns   [3.50/5.00]
         mood match: 'chill' (+2.0)
```

What happened: Groq Llama extracted `mood=moody` and `energy=0.30` from "rainy" and
"evening" context. Unlike the template response, the Groq narrative names specific songs,
explains *why* each fits, and uses conversational language — all grounded strictly in the
retrieved list. Retrieval was identical to the no-API path; only the narrative layer changed.

---

## 6. Design Decisions and Trade-offs

### Why RAG instead of a fine-tuned model?

A fine-tuned model would require a training dataset of (query → song) pairs that does
not exist. RAG lets the system use the existing catalog as its knowledge base and
generate responses grounded in real data — no training required, and adding songs to
`songs.csv` immediately improves recommendations without touching the model.

### Why keyword NLP as a fallback instead of requiring the API?

Making the API key mandatory would exclude anyone running the project locally without
a paid account, which is a significant barrier for a portfolio or classroom project.
The keyword fallback produces the same retrieval results as the Groq path — only the
narrative wording differs. This means the core RAG architecture works identically in
both modes; the fallback is not a degraded version, it is a different surface layer
on the same pipeline.

### Why keep genre as a weak signal (+1.0) instead of strengthening it?

Experimentation showed that increasing the genre weight to +2.0 caused the system to
surface off-mood songs just because they matched the genre. The original balance — mood
and energy together worth 4 of 5 points, genre worth 1 — correctly prioritizes how a
song *feels* over what it is labeled. Genre is a tiebreaker, not a primary filter,
which mirrors how real recommenders treat structural metadata.

### What was intentionally left unfixed?

The `likes_acoustic` user preference is collected but not used in scoring. This was a
deliberate choice to preserve a documented limitation from the base project for
pedagogical transparency — it shows that the system has a known gap. In a production
system this would be the first thing to fix.

### Trade-offs accepted:

| Decision | Benefit | Cost |
|---|---|---|
| Keyword fallback | No API key required | Extraction quality lower than Groq Llama |
| Exact mood string matching | Simple, explainable | "relaxed" and "chill" score 0 for each other |
| 18-song catalog | Fast, auditable | Most genres have only 1 song; thin coverage |
| Logging to file | Full audit trail | Log grows without rotation |

---

## 7. Testing Summary

> **Quick summary:** 11/11 pytest tests pass. The RAG pipeline evaluation scores 6/6
> functional tests with an average confidence of 0.92. One test intentionally documents
> a known limitation (mood-label vocabulary mismatch). The word-boundary bug fix
> (`"work"` firing inside `"working"`) was discovered and resolved during test
> development.

### How to run all tests

```bash
# From the project root
pytest tests/ -v
```

```
tests/test_ai_pipeline.py::test_study_session               PASSED
tests/test_ai_pipeline.py::test_late_night_drive            PASSED
tests/test_ai_pipeline.py::test_morning_run_pop             PASSED
tests/test_ai_pipeline.py::test_workout_high_energy         PASSED
tests/test_ai_pipeline.py::test_coffee_shop_jazz            PASSED
tests/test_ai_pipeline.py::test_melancholic_folk            PASSED
tests/test_ai_pipeline.py::test_confidence_score_empty      PASSED
tests/test_ai_pipeline.py::test_confidence_score_high_match PASSED
tests/test_ai_pipeline.py::test_known_failure_mood_label_trap PASSED
tests/test_recommender.py::test_recommend_returns_songs_sorted_by_score PASSED
tests/test_recommender.py::test_explain_recommendation_returns_non_empty_string PASSED

11 passed in 0.23s
```

### Evaluation harness (`tests/test_ai_pipeline.py`)

Run as a standalone script for a detailed report with confidence scores:

```bash
python tests/test_ai_pipeline.py
```

The harness runs 7 predefined (query → expected result) cases against the keyword-based
extractor. No API key is required.

| # | Query | Expected | Confidence | Result |
|---|---|---|---|---|
| 1 | "studying for finals, calm background" | mood=chill, lofi/ambient in top-2 | 0.79 | PASS |
| 2 | "moody late night drive" | top = Night Drive Loop | 0.89 | PASS |
| 3 | "happy upbeat pop morning run" | mood=happy, pop in top-1 | 1.00 | PASS |
| 4 | "aggressive intense metal workout" | energy >= 0.85 extracted and retrieved | 0.81 | PASS |
| 5 | "relaxed jazz coffee shop" | top = Coffee Shop Stories | 1.00 | PASS |
| 6 | "melancholic folk acoustic guitar" | top = Autumn Letters | 1.00 | PASS |
| 7 | "ambient relaxed background" | top genre = ambient *(known to fail)* | 0.94 | KNOWN FAIL |

**Average confidence score: 0.92** — interpreted as HIGH (most queries had a clear
best match with a significant score gap between #1 and #2).

### Confidence scoring

Each retrieval result gets a confidence score (0.0–1.0) computed as:

```
confidence = (top_score / 5.0) + min(gap_between_1st_and_2nd / 1.0, 1.0) × 0.15
```

| Range | Meaning |
|---|---|
| >= 0.80 | High — mood + genre + energy all aligned |
| 0.60–0.79 | Moderate — 1–2 signals matched |
| < 0.60 | Low — energy-only matches, catalog gap |

### Bug found during testing

During test development the keyword extractor incorrectly mapped `"working"` → `"work"`
(a substring match), which set `mood=focused` and `energy=0.45` for a workout query
instead of `mood=intense` and `energy=0.93`. The fix was switching all keyword matches
from `if kw in text` (substring) to `re.search(r"\b" + kw + r"\b", text)` (whole-word).
This is documented as the `_match()` helper in `src/ai_recommender.py`.

### Unit tests (`tests/test_recommender.py`) — original system

Two pytest tests cover the OOP interface:

1. `test_recommend_returns_songs_sorted_by_score` — top result for a pop/happy/high-energy
   user is the pop, happy song.
2. `test_explain_recommendation_returns_non_empty_string` — explanation method returns
   readable text.

Both pass and serve as regression guards if the scoring formula changes.

### Adversarial bias tests (`src/adversarial_test.py`) — 8 profiles

Human-reviewed tests designed to expose failure modes:

| Profile | What it tests | Result |
|---|---|---|
| Conflicting signals (blues/sad + energy=0.9) | High energy override mood? | Right song at #1, energy filler at #2–5 |
| Strict filter blackhole (classical + angry) | Silent empty result? | Empty list — no explanation shown |
| Mood label trap (ambient/relaxed) | "relaxed" != "chill" as strings? | Jazz ranked #1 over the ambient song |
| Energy floor (classical/peaceful, target=0.0) | Perfect score reachable? | Max 4.56 — no song has energy exactly 0.0 |
| Acousticness ghost (folk, likes_acoustic=False) | likes_acoustic used? | Top 5 all highly acoustic — field ignored |
| Tie ambush (lofi/focused, energy=0.60) | Energy float unrelated genres? | Country at #5 with no mood or genre match |
| Energy override (rock/sad, target=0.95) | Genre vs. mood conflict? | Genre and mood pull in opposite directions |
| Middle energy attractor (hip-hop/angry, target=0.5) | Median energy over-attract? | Metal #1, country #3, lofi #5 |

**What worked well:** Standard profiles returned expected songs. Every score is
auditable — any result can be explained in one line of arithmetic.

**What did not work well:** Semantic mood similarity (relaxed vs. chill), thin-catalog
genres (1 song = energy noise at positions 2–5), and the unused `likes_acoustic` field.

**What the tests taught:** Bias in a recommender often lives in the data labeling, not
the algorithm. It is invisible until you construct a case designed to find it.

---

## 8. Reflection

### What this project taught about AI

The single most important thing this project demonstrated is that an AI system can
produce outputs that *feel* intelligent without containing any intelligence. The three
standard profiles returned sensible, coherent recommendations — but the entire decision
was three numbers added together. The feeling of intelligence came from well-structured
data, not from the algorithm.

The RAG extension added a genuine AI layer, but it also revealed the same lesson at a
higher level. The Groq Llama narrative response sounds fluent and personalized. But
everything it says is constrained by what the retriever returned. If the retriever
surfaces a bad catalog match, the LLM will write a convincing paragraph explaining why
that bad match is actually good. The AI layer does not fix retrieval errors — it narrates
them. Retrieval quality is the ceiling; generation quality is the floor.

### What this project taught about bias

The ambient-gets-jazz failure is the clearest example. A user asked for ambient, relaxed
music. They received a jazz song first. Nothing in the code was wrong. The math was
correct. The bias came from a one-word labeling inconsistency in the data — the ambient
song was tagged "chill" and the user preference was "relaxed." They mean the same thing
in English. They score zero for each other as strings.

This kind of bias is nearly impossible to catch by reading the code. It only becomes
visible when you deliberately construct a case designed to find it. Systematic
adversarial testing is not optional in AI systems — it is how you find the failures that
the code will never show you.

### How AI tools were used during development

Claude was used during the development of this extension for three things:

1. **Generating adversarial test profiles** — Claude suggested edge cases like "combine
   sad mood with high energy" and "use strict filters for two categories that don't
   overlap." These were more creative and probing than the profiles I would have
   written alone, and several of them found real failures.

2. **Drafting the keyword-to-mood mapping table** — Claude generated a first draft of
   the `_MOOD_KEYWORDS` and `_ENERGY_HINTS` dictionaries. This was useful but required
   manual review — Claude mapped "blues" as both a genre and a mood synonym for "sad,"
   which was correct musically but conflated two separate fields in the data model.
   That required a careful fix.

3. **Debugging the Unicode encoding error** — When the `→` arrow character caused a
   `UnicodeEncodeError` on Windows (cp1252 encoding), Claude immediately identified the
   cause and proposed replacing it with `->`. This was correct and fast.

**One AI suggestion that was helpful:** The suggestion to add a dedicated guardrail that
clamped extracted energy to `[0.0, 1.0]` after the Groq function call response, because LLMs
can occasionally output slightly out-of-range numbers (e.g., 1.05) even with a defined
schema. Adding that clamp made the system robust to a real failure mode.

**One AI suggestion that required correction:** The initial suggestion put the provider
import at the top of `ai_recommender.py` unconditionally. This caused an `ImportError`
for users who had not installed the package. The fix was to import it lazily inside the
function branch that actually uses it, so the keyword-only path never touches the import.

### Limitations and what comes next

The biggest limitation is catalog size. With 18 songs and 15 genres, most genres have
exactly one representative. Any user whose preferences land in an underrepresented genre
will receive energy-sorted noise for positions 2 through 5. No amount of algorithmic
tuning fixes this — it requires more songs.

The second limitation is that mood matching is still all-or-nothing in the scoring
engine. The RAG layer helps users express themselves more naturally, but once the
keyword extractor maps "relaxed" to a catalog mood label, the exact-string matching
problem reappears. A semantic similarity layer (even cosine distance on mood word
embeddings) would fix this without requiring a larger model.

If this project were taken further, the highest-value next step would be grouping
semantically similar moods (chill, relaxed, peaceful → one cluster) and scoring partial
credit for near-matches — not because it is technically interesting, but because that
single change would fix more user-facing failures than any other improvement.

---

## 9. Responsible AI Reflection

### What are the limitations or biases in this system?

A few things stood out as I built and tested this.

The biggest one is **vocabulary bias**. The scorer only gives points for exact string
matches, so "relaxed" and "chill" score zero for each other even though they mean the
same thing. A user's experience depends more on whether their words happen to match the
catalog's labels than on whether the music actually fits what they wanted. That's not
fixable by tweaking weights — it needs a semantic layer like mood groupings or synonyms.

There's also a **catalog fairness problem**. Lofi has 3 songs, but hip-hop, blues, and
classical each have 1. Lofi users get genuinely good results. Everyone else gets one
real match and four energy-sorted fillers. The system quietly favors whoever matches an
overrepresented genre, which would get worse over time in a real product as
well-served users keep engaging and under-served users leave.

One I didn't expect: the **AI narrative sounds confident even when it's wrong**. During
testing, the mood-label trap case scored 0.94 confidence and got a fluent, friendly
response — but returned jazz when the user asked for ambient. A polished AI voice can
make a wrong answer feel right, which is arguably more harmful than just getting it wrong
silently.

---

### Could this AI be misused? How would you prevent it?

This specific tool is pretty low-stakes — the worst case is a bad playlist. But the
underlying pattern is worth thinking about.

The RAG pipeline generates persuasive, personalized narratives around whatever it
retrieves. Apply the same architecture to medical advice or financial recommendations
and the fluency of the response would actively suppress the user's skepticism even when
the retrieved content is wrong. The defense I built in here — always showing the raw
ranked list and scores alongside the narrative — is the right instinct: give users
enough information to push back on the AI rather than just trust it.

A smaller risk is **catalog injection**: if the song list weren't fixed, someone could
add a song tagged to match every mood and genre and it would reliably show up first.
That's the same problem as SEO spam, just applied to metadata. Treating the catalog as
trusted, version-controlled content is the prevention.

---

### What surprised you while testing reliability?

Two things genuinely caught me off guard.

First, the **word-boundary bug was completely invisible** until an automated test found
it. The extractor matched `"work"` inside `"working"` and silently set mood=focused,
energy=0.45 for a workout query — the output still produced a recommendation, just the
wrong kind of music. No error, no warning, nothing obviously wrong at a glance. I
wouldn't have caught it without a test that actually checked the extracted values. The
lesson: a system that fails quietly is harder to debug than one that crashes loudly.

Second, **high confidence doesn't mean correct**. The mood-label trap case had the
second-highest confidence score in the whole suite (0.94) and returned the wrong song.
Confidence only measures how clearly one result beats the others — it doesn't know if
any of them were actually good. That flipped how I think about the scores: useful for
flagging low-quality results, but not trustworthy as a stamp of approval.

---

### AI Collaboration: one helpful suggestion, one flawed one

**Helpful — the energy guardrail.** Claude suggested clamping the extracted energy value
to `[0.0, 1.0]` even though the tool schema already defined those as the min/max. The
reasoning was that LLMs occasionally produce out-of-range numbers like `1.05` even with
a schema constraint because they're generating tokens, not enforcing rules. That turned
out to be exactly right, and the one-line fix (`max(0.0, min(1.0, val))`) cost nothing.
It was the kind of heads-up that comes from real experience with how these models
actually behave in practice.

**Flawed — the unconditional import.** The initial suggestion put the provider import at
the top of `ai_recommender.py`, which is normal Python style. But it meant anyone
without the `groq` package installed got an `ImportError` even if they only wanted the
keyword-based path that never touches the API. The fix was lazy importing — moving it
inside the function that actually uses it. It's a common pattern for optional
dependencies that the AI missed because it defaulted to the most standard convention
without thinking about whether the import was required or optional. Small thing, but a
good reminder to review dependency handling yourself rather than assuming the AI caught
all the edge cases.

---

## Project Structure

```
Applied-AI-system-project/
├── app.py                     # Streamlit UI — visual interface wrapping the pipeline
├── data/
│   └── songs.csv              # 18-song catalog, 10 audio features per song
├── src/
│   ├── main.py                # CLI entry point, mode selector
│   ├── recommender.py         # Core scoring algorithm (OOP + functional interfaces)
│   ├── ai_recommender.py      # RAG pipeline: extractor, retriever, generator, guardrails
│   └── adversarial_test.py    # 8 edge-case bias-detection profiles
├── tests/
│   ├── test_recommender.py    # pytest unit tests (OOP interface)
│   └── test_ai_pipeline.py    # pytest + standalone eval harness (7 query→expected cases)
├── model_card.md              # Model evaluation, limitations, bias documentation
├── reflection.md              # 5 comparative profile pair analyses
├── requirements.txt           # groq, pandas, pytest, python-dotenv, streamlit
└── .env.example               # API key template
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `groq` | latest | Groq Llama API (optional — keyword fallback used if absent) |
| `python-dotenv` | latest | Loads `GROQ_API_KEY` from `.env` file automatically |
| `pandas` | latest | CSV loading utilities |
| `pytest` | latest | Unit test runner |
| `streamlit` | latest | Web UI — served by `streamlit run app.py` |

See [model_card.md](model_card.md) for a detailed breakdown of known limitations and
bias patterns.