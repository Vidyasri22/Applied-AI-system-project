"""
Evaluation harness for the RAG pipeline.

Runs 7 predefined (natural-language query -> expected output) test cases using
the keyword-based extractor (no API key required) and reports:
  - pass / fail per test
  - confidence score per retrieval
  - overall summary line

One test is marked expect_fail=True to document a known system limitation
(mood-label vocabulary mismatch: "relaxed" != "chill" as strings).

Usage
-----
  # as a standalone script from the project root:
  python tests/test_ai_pipeline.py

  # or via pytest:
  pytest tests/test_ai_pipeline.py -v
"""

import sys
import os

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import load_songs
from ai_recommender import _keyword_extract, retrieve_candidates, confidence_score

_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

# ---------------------------------------------------------------------------
# Test cases
# Each case has:
#   query       - natural-language input string
#   label       - human-readable test name
#   description - what the check verifies
#   check       - callable(prefs, candidates) -> bool
#   expect_fail - True when the test documents a known limitation
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": 1,
        "label": "Study session — calm/lofi",
        "query": "studying for finals and need calm background music",
        "description": "mood=chill, top-2 results contain a lofi or ambient song",
        "check": lambda prefs, cands: (
            prefs["favorite_mood"] == "chill"
            and any(s["genre"] in ("lofi", "ambient") for s, _, _ in cands[:2])
        ),
        "expect_fail": False,
    },
    {
        "id": 2,
        "label": "Late night drive — moody",
        "query": "moody atmospheric music for a late night drive",
        "description": "top song is Night Drive Loop",
        "check": lambda prefs, cands: cands[0][0]["title"] == "Night Drive Loop",
        "expect_fail": False,
    },
    {
        "id": 3,
        "label": "Morning run — happy/pop",
        "query": "happy upbeat pop song for a morning run",
        "description": "mood=happy, top song genre is pop or indie pop",
        "check": lambda prefs, cands: (
            prefs["favorite_mood"] == "happy"
            and cands[0][0]["genre"] in ("pop", "indie pop")
        ),
        "expect_fail": False,
    },
    {
        "id": 4,
        "label": "Workout — high energy",
        "query": "aggressive intense metal for working out",
        "description": "extracted energy >= 0.85, top song energy >= 0.85",
        "check": lambda prefs, cands: (
            prefs["target_energy"] >= 0.85
            and cands[0][0]["energy"] >= 0.85
        ),
        "expect_fail": False,
    },
    {
        "id": 5,
        "label": "Coffee shop — relaxed/jazz",
        "query": "relaxed jazz for a quiet coffee shop afternoon",
        "description": "top song is Coffee Shop Stories",
        "check": lambda prefs, cands: cands[0][0]["title"] == "Coffee Shop Stories",
        "expect_fail": False,
    },
    {
        "id": 6,
        "label": "Melancholic folk — acoustic",
        "query": "melancholic folk music with acoustic guitar",
        "description": "top song is Autumn Letters",
        "check": lambda prefs, cands: cands[0][0]["title"] == "Autumn Letters",
        "expect_fail": False,
    },
    {
        "id": 7,
        "label": "KNOWN FAILURE — Mood label trap (ambient/relaxed)",
        "query": "ambient relaxed background music",
        "description": (
            "top song genre is ambient — EXPECTED TO FAIL: "
            "'relaxed' != 'chill' string match returns jazz first"
        ),
        "check": lambda prefs, cands: cands[0][0]["genre"] == "ambient",
        "expect_fail": True,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_evaluation(songs: list) -> None:
    passed = 0
    expected_failures = 0
    total = len(TEST_CASES)
    confidence_scores = []

    print("\n" + "=" * 66)
    print("  RAG Pipeline Evaluation — keyword extraction (no API key)")
    print("=" * 66)

    for case in TEST_CASES:
        prefs = _keyword_extract(case["query"])
        candidates = retrieve_candidates(prefs, songs, k=5)
        conf = confidence_score(candidates)
        confidence_scores.append(conf)

        try:
            result = case["check"](prefs, candidates)
        except (IndexError, KeyError):
            result = False

        top_title = candidates[0][0]["title"] if candidates else "—"
        top_score = candidates[0][1] if candidates else 0.0

        if case["expect_fail"]:
            # Known failure: pass the harness if the check fails (documenting the bug)
            harness_pass = not result
            status = "KNOWN FAIL (documented)" if harness_pass else "UNEXPECTED PASS"
            expected_failures += 1
        else:
            harness_pass = result
            status = "PASS" if harness_pass else "FAIL"
            if harness_pass:
                passed += 1

        print(f"\n  [{case['id']}] {case['label']}")
        print(f"       Query:      \"{case['query']}\"")
        print(f"       Extracted:  mood={prefs['favorite_mood']}, "
              f"genre={prefs['favorite_genre']}, energy={prefs['target_energy']:.2f}")
        print(f"       Top result: \"{top_title}\"  (score {top_score:.2f}/5.00)")
        print(f"       Confidence: {conf:.2f}")
        print(f"       Check:      {case['description']}")
        print(f"       Status:     {status}")

    avg_conf = sum(confidence_scores) / len(confidence_scores)
    regular_total = total - sum(1 for c in TEST_CASES if c["expect_fail"])

    print("\n" + "=" * 66)
    print(f"  RESULTS: {passed}/{regular_total} tests passed  "
          f"| {expected_failures} known failure(s) confirmed")
    print(f"  Average confidence score: {avg_conf:.2f}")

    if avg_conf >= 0.70:
        conf_label = "HIGH — most queries had a clear best match"
    elif avg_conf >= 0.50:
        conf_label = "MODERATE — some queries lacked a strong catalog match"
    else:
        conf_label = "LOW — catalog coverage is thin for several query types"
    print(f"  Confidence interpretation: {conf_label}")
    print("=" * 66 + "\n")

    return passed, regular_total, avg_conf


# ---------------------------------------------------------------------------
# pytest-compatible test functions (also called by __main__ below)
# ---------------------------------------------------------------------------
def _load():
    return load_songs(_CSV)


def test_study_session():
    songs = _load()
    prefs = _keyword_extract("studying for finals and need calm background music")
    cands = retrieve_candidates(prefs, songs)
    assert prefs["favorite_mood"] == "chill"
    assert any(s["genre"] in ("lofi", "ambient") for s, _, _ in cands[:2])


def test_late_night_drive():
    songs = _load()
    prefs = _keyword_extract("moody atmospheric music for a late night drive")
    cands = retrieve_candidates(prefs, songs)
    assert cands[0][0]["title"] == "Night Drive Loop"


def test_morning_run_pop():
    songs = _load()
    prefs = _keyword_extract("happy upbeat pop song for a morning run")
    cands = retrieve_candidates(prefs, songs)
    assert prefs["favorite_mood"] == "happy"
    assert cands[0][0]["genre"] in ("pop", "indie pop")


def test_workout_high_energy():
    songs = _load()
    prefs = _keyword_extract("aggressive intense metal for working out")
    cands = retrieve_candidates(prefs, songs)
    assert prefs["target_energy"] >= 0.85
    assert cands[0][0]["energy"] >= 0.85


def test_coffee_shop_jazz():
    songs = _load()
    prefs = _keyword_extract("relaxed jazz for a quiet coffee shop afternoon")
    cands = retrieve_candidates(prefs, songs)
    assert cands[0][0]["title"] == "Coffee Shop Stories"


def test_melancholic_folk():
    songs = _load()
    prefs = _keyword_extract("melancholic folk music with acoustic guitar")
    cands = retrieve_candidates(prefs, songs)
    assert cands[0][0]["title"] == "Autumn Letters"


def test_confidence_score_empty():
    assert confidence_score([]) == 0.0


def test_confidence_score_high_match():
    # Simulate a near-perfect match (score 4.9) with a clear gap
    mock = [
        ({"title": "A"}, 4.9, ""),
        ({"title": "B"}, 2.0, ""),
    ]
    score = confidence_score(mock)
    assert score >= 0.80, f"Expected >= 0.80 for strong match, got {score}"


def test_known_failure_mood_label_trap():
    """
    Documents the known mood-label vocabulary bug.
    'relaxed' != 'chill' as exact strings, so the ambient/relaxed user gets
    a jazz song at #1 instead of the actual ambient song.
    This test ASSERTS the bug is present — it passes when the bug exists.
    """
    songs = _load()
    prefs = _keyword_extract("ambient relaxed background music")
    cands = retrieve_candidates(prefs, songs)
    # The bug: top result is NOT ambient even though user asked for ambient
    assert cands[0][0]["genre"] != "ambient", (
        "Mood label trap no longer fires — the bug may have been fixed. "
        "Update this test if partial mood matching was added."
    )


if __name__ == "__main__":
    songs = load_songs(_CSV)
    run_evaluation(songs)