"""
MoodMatch — Streamlit UI for the RAG-powered music recommender.

Run:   streamlit run app.py
"""

import os
import sys

import streamlit as st

# Make src/ importable so we can reuse the existing pipeline
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from recommender import load_songs, score_song  # noqa: E402
from ai_recommender import (  # noqa: E402
    _keyword_extract,
    _groq_extract,
    _groq_generate,
    _template_response,
    retrieve_candidates,
    confidence_score,
)

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MoodMatch — AI Music Recommender",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Subtle typography + spacing polish */
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

    /* Card-like containers */
    .song-card {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(236,72,153,0.04));
    }
    .song-card-top {
        border: 1px solid rgba(99,102,241,0.35);
        background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(236,72,153,0.08));
    }
    .rank-badge {
        display: inline-block;
        background: #6366f1;
        color: white;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .rank-badge.gold { background: linear-gradient(135deg, #f59e0b, #ec4899); }
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(99,102,241,0.12);
        color: #6366f1;
        margin-right: 6px;
        font-weight: 500;
    }
    .narrative-box {
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        background: rgba(99,102,241,0.05);
        border-radius: 6px;
        line-height: 1.6;
        font-size: 1.02rem;
    }
    .guardrail-fired {
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        background: rgba(245,158,11,0.08);
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .mode-badge-groq {
        background: linear-gradient(135deg, #10b981, #6366f1);
        color: white; padding: 4px 12px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .mode-badge-kw {
        background: rgba(128,128,128,0.15);
        color: inherit; padding: 4px 12px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .muted { color: rgba(128,128,128,0.85); font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Catalog + Groq client (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_catalog():
    csv_path = os.path.join(_ROOT, "data", "songs.csv")
    return load_songs(csv_path)


@st.cache_resource(show_spinner=False)
def _load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "keyword"
    try:
        from groq import Groq
        return Groq(api_key=api_key), "groq"
    except ImportError:
        return None, "keyword"


songs = _load_catalog()
client, mode = _load_groq_client()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### MoodMatch")
    st.caption("RAG-powered music recommender")

    if mode == "groq":
        st.markdown(
            '<span class="mode-badge-groq">Groq Llama 3.3 70B</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="mode-badge-kw">Keyword NLP (no API key)</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Catalog**")
    st.caption(f"{len(songs)} songs across 15 genres")

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        """
        1. **Extract** preferences from your sentence
        2. **Retrieve** top-5 songs from the catalog
        3. **Generate** a grounded narrative

        Every song named in the output is a real row
        in `data/songs.csv` — the LLM cannot hallucinate.
        """
    )

    st.markdown("---")
    st.markdown("**Guardrails active**")
    st.markdown(
        """
        - Energy clamped to `[0.0, 1.0]`
        - Missing key → keyword fallback
        - Empty retrieval → skip narrative
        - Whole-word regex on keywords
        """
    )

    with st.expander("GitHub & Loom"):
        st.markdown("[GitHub repo](https://github.com/Vidyasri22/Applied-AI-system-project)")
        st.caption("Loom walkthrough: see README")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("MoodMatch")
st.markdown(
    '<p class="muted">Describe what you want to listen to in plain English. '
    "The system extracts your preferences, retrieves the best-matching songs from "
    "an 18-song catalog, and writes a recommendation grounded only in what was retrieved.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input row
# ---------------------------------------------------------------------------
EXAMPLES = [
    "I'm studying for finals and need something calm but focused",
    "I need something moody for a late night drive",
    "something really aggressive and angry for the gym",
    "I want something moody and atmospheric for a rainy evening",
]

if "query" not in st.session_state:
    st.session_state.query = EXAMPLES[0]

st.markdown("#### What kind of music are you looking for?")

query = st.text_area(
    label="query",
    value=st.session_state.query,
    height=80,
    label_visibility="collapsed",
    placeholder="e.g. I need something moody for a late night drive",
    key="query_input",
)

col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
for col, ex in zip((col_ex1, col_ex2, col_ex3, col_ex4), EXAMPLES):
    with col:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:10]}"):
            st.session_state.query = ex
            st.rerun()

st.markdown("")
go = st.button("Recommend music", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _run_pipeline(user_input: str):
    guardrails_fired = []

    if client is not None:
        prefs = _groq_extract(user_input, client)
        if prefs is None:
            prefs = _keyword_extract(user_input)
            guardrails_fired.append("Groq extraction failed — fell back to keyword NLP")
    else:
        prefs = _keyword_extract(user_input)

    raw_energy = prefs["target_energy"]
    clamped = max(0.0, min(1.0, float(raw_energy)))
    if clamped != raw_energy:
        guardrails_fired.append(f"Energy clamped from {raw_energy} to {clamped}")
    prefs["target_energy"] = clamped

    candidates = retrieve_candidates(prefs, songs, k=5)

    if not candidates:
        guardrails_fired.append("No songs retrieved — narrative step skipped")
        return prefs, candidates, "", 0.0, guardrails_fired

    if client is not None:
        narrative = _groq_generate(user_input, prefs, candidates, client)
    else:
        narrative = _template_response(user_input, prefs, candidates)

    conf = confidence_score(candidates)
    top_score = candidates[0][1]
    if top_score < 2.5:
        guardrails_fired.append(f"Low top score ({top_score:.2f}/5.00) — catalog coverage gap flagged")

    return prefs, candidates, narrative, conf, guardrails_fired


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if go and query.strip():
    with st.spinner("Extracting preferences → Retrieving → Generating..."):
        prefs, candidates, narrative, conf, guardrails = _run_pipeline(query.strip())

    st.markdown("---")

    st.markdown("### Extracted preferences")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genre", prefs["favorite_genre"].title())
    m2.metric("Mood", prefs["favorite_mood"].title())
    m3.metric("Energy", f"{prefs['target_energy']:.2f}")
    m4.metric("Confidence", f"{conf:.2f}", help="Combines top score and gap to #2")

    if prefs.get("reasoning"):
        st.caption(f"Reasoning: {prefs['reasoning']}")

    if guardrails:
        st.markdown("### Guardrails")
        for g in guardrails:
            st.markdown(f'<div class="guardrail-fired"><b>Fired:</b> {g}</div>',
                        unsafe_allow_html=True)

    if not candidates:
        st.warning("Nothing in the catalog matched this request. Try different keywords.")
    else:
        st.markdown("### Recommendation")
        st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

        st.markdown("### Ranked results")
        for rank, (song, score, _) in enumerate(candidates, start=1):
            _, reasons = score_song(prefs, song)
            signal_pills = "".join(
                f'<span class="pill">{r}</span>'
                for r in reasons
                if not r.startswith("energy proximity")
            )
            if not signal_pills:
                signal_pills = '<span class="pill">energy proximity only</span>'

            card_class = "song-card song-card-top" if rank == 1 else "song-card"
            badge_class = "rank-badge gold" if rank == 1 else "rank-badge"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span class="{badge_class}">#{rank}</span>
                            <b style="font-size:1.05rem;">{song['title']}</b>
                            <span class="muted">by {song['artist']}</span>
                        </div>
                        <div style="font-family:monospace; font-size:0.95rem;">
                            {score:.2f} / 5.00
                        </div>
                    </div>
                    <div style="margin-top:8px;">
                        <span class="pill">{song['genre']}</span>
                        <span class="pill">{song['mood']}</span>
                        <span class="pill">energy {song['energy']:.2f}</span>
                    </div>
                    <div style="margin-top:8px;">
                        {signal_pills}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(score / 5.0, 1.0))

        with st.expander("See raw retrieval details"):
            st.json({
                "query": query.strip(),
                "extracted_preferences": prefs,
                "confidence": conf,
                "num_candidates": len(candidates),
                "top_song": {
                    "title": candidates[0][0]["title"],
                    "artist": candidates[0][0]["artist"],
                    "score": candidates[0][1],
                },
                "guardrails_fired": guardrails,
            })
else:
    st.info("Pick an example above or type your own query, then click **Recommend music**.")
