"""
app.py – Epstein Investigation  |  Streamlit Dashboard
=======================================================
Loads the pipeline artefacts produced by ``main.py`` and exposes a
three-tab interactive dashboard:

  Tab 1 – Dashboard       : corpus stats, risk distribution, offense breakdown
  Tab 2 – Email Explorer  : multi-keyword / filter search + full email detail view
  Tab 3 – Semantic Search : FAISS-powered nearest-neighbour retrieval

Run
---
    streamlit run app.py
"""

import ast
import os
import re

import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH        = "data/classified_emails.csv"
EMBEDDINGS_PATH  = "data/embeddings.npy"
EMBEDDER_MODEL   = "all-MiniLM-L6-v2"
BINARY_THRESHOLD = 0.35

OFFENSE_CATEGORIES = [
    "Sexual exploitation or trafficking",
    "Financial fraud or money laundering",
    "Obstruction or witness tampering",
    "Bribery or corruption",
    "Coercion or blackmail",
    "Network facilitation or coordination",
]

PALETTE = {
    "problematic": "#C0392B",
    "safe":        "#2E5FA3",
    "neutral":     "#7F8C8D",
    "accent":      "#E67E22",
}

# ── Name normalisation ────────────────────────────────────────────────────────
# Entities that spaCy sometimes tags as PERSON but are not people
_NON_PERSON_TOKENS = frozenset({
    # Social media & tech apps
    "twitter", "facebook", "instagram", "google", "youtube", "snapchat",
    "whatsapp", "linkedin", "tiktok", "apple", "microsoft", "amazon",
    "netflix", "uber", "lyft", "paypal", "venmo", "blackberry", "nokia",
    "telegram", "signal", "skype", "zoom", "slack",
    # News & media outlets
    "cnn", "bbc", "nbc", "abc", "cbs", "msnbc", "fox", "fox news",
    "new york times", "washington post", "reuters", "ap",
    "associated press", "daily mail", "new york post",
    # Government agencies & abbreviations
    "fbi", "cia", "nsa", "doj", "sec", "dea", "nypd", "interpol",
    "u.s.", "u.k.", "usa", "america",
    # Honorifics that get extracted on their own
    "mr", "ms", "mrs", "dr", "sir", "lord", "lady", "hon", "esq",
})

PERSON_ALIAS_GROUPS = {
    "Donald Trump": {
        "donald",
        "trump",
        "donald trump",
    },
    "Jeffrey Epstein": {
        "epstein",
        "jeffrey",
        "jeffrey epstein",
    },
    "Bill Clinton": {
        "bill",
        "clinton",
        "bill clinton",
    },
}

PERSON_ALIAS_LOOKUP = {
    alias.lower(): canonical
    for canonical, aliases in PERSON_ALIAS_GROUPS.items()
    for alias in aliases
}


def _normalize_name(name: str) -> str:
    """Title-case a person name consistently."""
    return " ".join(
        "-".join(part.capitalize() for part in word.split("-"))
        for word in name.strip().split()
    )


def _is_valid_person(name: str) -> bool:
    """Return False for names that are clearly not real people."""
    n = name.strip().lower()
    return (
        n not in _NON_PERSON_TOKENS
        and len(n) >= 3
        and sum(c.isdigit() for c in n) <= 1
    )


def normalize_persons(persons: list[str]) -> list[str]:
    """
    Normalize person names using explicit alias groups first,
    then fall back to cleaned title-cased names.
    """
    normalized = []

    for person in persons:
        if not isinstance(person, str):
            continue

        raw = person.strip()
        if not raw or not _is_valid_person(raw):
            continue

        key = raw.lower().strip()

        # Explicit grouping first
        if key in PERSON_ALIAS_LOOKUP:
            normalized.append(PERSON_ALIAS_LOOKUP[key])
        else:
            normalized.append(_normalize_name(raw))

    return normalized

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epstein Investigation Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
        .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
        .detail-box {
            background: #f8f9fa;
            border-left: 4px solid #C0392B;
            padding: 1rem 1.2rem;
            border-radius: 4px;
            font-size: 0.88rem;
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading classified emails …")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Restore Python list columns that were serialised as strings
    for col in ("senders", "offense_labels"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: (
                    ast.literal_eval(x)
                    if isinstance(x, str) and x.startswith("[")
                    else x
                )
            )

    # Ensure derived display columns exist
    if "offense_labels_display" not in df.columns:
        df["offense_labels_display"] = df["offense_labels"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    if "senders_display" not in df.columns:
        df["senders_display"] = df["senders"].apply(
            lambda x: ", ".join(x[:3]) if isinstance(x, list) else str(x)
        )
    if "risk_display" not in df.columns:
        df["risk_display"] = df["risk_flag"].map(
            {1: "⚠️ Problematic", 0: "✅ Safe"}
        )
    if "thread_id" not in df.columns:
        df["thread_id"] = df.index.astype(str)
    if "message_count" not in df.columns:
        df["message_count"] = 1

    return df


@st.cache_resource(show_spinner="Loading semantic search index …")
def load_search_index():
    """Build a FAISS inner-product index from pre-computed embeddings."""
    embedder   = SentenceTransformer(EMBEDDER_MODEL)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return embedder, index


# ── Chart builders ────────────────────────────────────────────────────────────

def make_risk_pie(df: pd.DataFrame) -> go.Figure:
    n_flagged = int(df["risk_flag"].sum())
    n_safe    = len(df) - n_flagged

    fig = go.Figure(data=[go.Pie(
        labels=["⚠️ Problematic", "✅ Non-Problematic"],
        values=[n_flagged, n_safe],
        hole=0.45,
        marker_colors=[PALETTE["problematic"], PALETTE["safe"]],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    )])
    fig.update_layout(
        title="Risk Distribution",
        template="plotly_white",
        height=360,
        margin=dict(t=75, b=20, l=20, r=20),
        showlegend=True,
    )
    return fig


def make_offense_bar(df: pd.DataFrame) -> go.Figure:
    offense_counts: dict = {}
    for labels in df[df["risk_flag"] == 1]["offense_labels"]:
        if isinstance(labels, list):
            for lab in labels:
                if lab not in ("none", "unclassified"):
                    offense_counts[lab] = offense_counts.get(lab, 0) + 1

    if not offense_counts:
        fig = go.Figure()
        fig.update_layout(
            title="No flagged emails to display",
            template="plotly_white",
            height=360,
        )
        return fig

    s = pd.Series(offense_counts).sort_values(ascending=True)
    fig = px.bar(
        x=s.values,
        y=s.index,
        orientation="h",
        title="Offense Category Breakdown",
        color_discrete_sequence=[PALETTE["problematic"]],
        labels={"x": "Count", "y": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def make_score_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="prob_score",
        nbins=30,
        color="risk_display",
        title="Risk Score Distribution",
        barmode="overlay",
        opacity=0.78,
        color_discrete_map={
            "⚠️ Problematic": PALETTE["problematic"],
            "✅ Safe":         PALETTE["safe"],
        },
        labels={"prob_score": "P(Problematic)", "risk_display": "Classification"},
    )
    fig.add_vline(
        x=BINARY_THRESHOLD,
        line_dash="dash",
        line_color=PALETTE["neutral"],
        annotation_text=f"Threshold = {BINARY_THRESHOLD}",
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def make_top_persons_bar(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of the most frequently mentioned *people* in flagged emails.
    Names are normalised (title-cased, deduplicated variants grouped) and
    non-person tokens (apps, agencies, etc.) are filtered out.
    """
    raw_persons: list[str] = []
    for entities_str in df[df["risk_flag"] == 1].get(
        "entities_json", pd.Series(dtype=str)
    ):
        try:
            ents = ast.literal_eval(str(entities_str))
            raw_persons.extend(ents.get("PERSON", []))
        except Exception:
            pass

    if not raw_persons:
        fig = go.Figure()
        fig.update_layout(title="No person entities found", template="plotly_white")
        return fig

    # Normalise and group name variants before counting
    normalised = normalize_persons(raw_persons)

    top = (
        pd.Series(normalised)
        .value_counts()
        .head(15)
        .sort_values(ascending=True)
    )

    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation="h",
        title="Top 15 People Mentioned in Flagged Emails",
        color_discrete_sequence=[PALETTE["neutral"]],
        labels={"x": "Mentions", "y": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


# ── Search helpers ────────────────────────────────────────────────────────────

def keyword_search(
    df: pd.DataFrame,
    keywords: list[str],
    match_mode: str,
    risk_filter: str,
    offense_filter: str,
    max_results: int,
) -> tuple[pd.DataFrame, list]:
    """
    Filter emails by one or more keywords, risk level, and offense category.

    Parameters
    ----------
    keywords : list[str]
        One or more search terms. Each term is matched independently against
        the email subject and body (case-insensitive).
    match_mode : str
        ``"AND"``  – email must contain **all** keywords (intersection).
        ``"OR"``   – email must contain **at least one** keyword (union).
    """
    results = df.copy()

    active = [kw for kw in keywords if kw.strip()]
    if active:
        if match_mode == "AND":
            # Start with all True, narrow down for each keyword
            mask = pd.Series(True, index=results.index)
            for kw in active:
                kw_mask = (
                    results["full_text"].str.contains(kw, case=False, na=False)
                    | results["subject"].str.contains(kw, case=False, na=False)
                )
                mask = mask & kw_mask
        else:  # OR
            # Start with all False, expand for each keyword
            mask = pd.Series(False, index=results.index)
            for kw in active:
                kw_mask = (
                    results["full_text"].str.contains(kw, case=False, na=False)
                    | results["subject"].str.contains(kw, case=False, na=False)
                )
                mask = mask | kw_mask

        results = results[mask]

    if risk_filter == "Problematic Only":
        results = results[results["risk_flag"] == 1]
    elif risk_filter == "Safe Only":
        results = results[results["risk_flag"] == 0]

    if offense_filter != "All":
        results = results[
            results["offense_labels_display"].str.contains(
                offense_filter, case=False, na=False
            )
        ]

    results = results.sort_values("prob_score", ascending=False).head(max_results)

    display = results[
        ["thread_id", "subject", "risk_display", "prob_score",
         "offense_labels_display", "senders_display", "date_range"]
    ].copy()
    display.columns = [
        "Thread ID", "Subject", "Risk", "Score",
        "Offense Categories", "Senders", "Date Range",
    ]
    display["Score"] = display["Score"].round(3)
    return display, results.index.tolist()


def semantic_search(
    query: str,
    df: pd.DataFrame,
    embedder,
    index,
    k: int = 10,
) -> pd.DataFrame:
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, indices = index.search(q_emb, k)

    rows = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(df):
            row = df.iloc[idx]
            rows.append({
                "Thread ID":  str(row["thread_id"]),
                "Subject":    row["subject"],
                "Risk":       row["risk_display"],
                "Similarity": round(float(score), 4),
                "Offense":    row["offense_labels_display"],
                "Preview":    str(row["full_text"])[:220] + " …",
            })
    return pd.DataFrame(rows)


# ── Email detail view ─────────────────────────────────────────────────────────

def render_email_detail(row: pd.Series) -> None:
    """Render a single email thread's full detail inside an expander."""
    risk_colour = PALETTE["problematic"] if row["risk_flag"] == 1 else PALETTE["safe"]
    badge = "⚠️ PROBLEMATIC" if row["risk_flag"] == 1 else "✅ SAFE"

    st.markdown(
        f"""
        <div class="detail-box">
          <b>Thread ID:</b> {row['thread_id']}<br>
          <b>Subject:</b> {row['subject']}<br>
          <b>Risk:</b> <span style="color:{risk_colour};font-weight:700">{badge}</span>
            &nbsp; (score: {row['prob_score']:.4f})<br>
          <b>Offense:</b> {row['offense_labels_display']}<br>
          <b>Senders:</b> {row['senders_display']}<br>
          <b>Date range:</b> {row.get('date_range', '—')}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Full Text**")
    st.text_area(
        label="",
        value=str(row["full_text"]),
        height=280,
        disabled=True,
        key=f"detail_{row['thread_id']}",
    )

    # Named entities
    entities_raw = row.get("entities_json") or row.get("entities", "{}")
    try:
        ents = ast.literal_eval(str(entities_raw))
        if ents:
            st.markdown("**Named Entities**")
            cols = st.columns(len(ents))
            for i, (label, values) in enumerate(ents.items()):
                cols[i].markdown(f"**{label}**")
                # Normalise person names in the detail view too
                if label == "PERSON":
                    values = list(dict.fromkeys(normalize_persons(values)))
                cols[i].write("\n".join(f"• {v}" for v in values[:10]))
    except Exception:
        pass


# ── Main application ──────────────────────────────────────────────────────────

def main() -> None:
    st.title("🔍 Epstein Email Investigation Dashboard")

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        st.error(
            "**Pipeline output not found.**  "
            "Run the classification pipeline first:\n\n"
            "```bash\npython main.py\n```"
        )
        st.stop()

    df = load_data()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("extras/logo.png", width=180)

        st.header("Dataset Summary")
        st.metric("Total Threads",     f"{len(df):,}")
        st.metric("Flagged",           f"{int(df['risk_flag'].sum()):,}")
        st.metric("Safe",              f"{len(df) - int(df['risk_flag'].sum()):,}")
        st.metric("Flagged %",         f"{100 * df['risk_flag'].mean():.1f} %")
        avg_msgs = df["message_count"].mean() if "message_count" in df.columns else 0
        st.metric("Avg Msgs / Thread", f"{avg_msgs:.1f}")

        st.divider()
        st.caption(
            "**Model:** facebook/bart-large-mnli  \n"
            "**Binary threshold:** 0.35  \n"
            "**Offense threshold:** 0.30  \n"
            "**Embeddings:** all-MiniLM-L6-v2"
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📊  Dashboard", "🔎  Email Explorer", "🧠  Semantic Search"]
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Dashboard
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Corpus Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Threads", f"{len(df):,}")
        c2.metric("Flagged",       f"{int(df['risk_flag'].sum()):,}")
        c3.metric("Safe",          f"{len(df) - int(df['risk_flag'].sum()):,}")
        c4.metric("Flagged %",     f"{100 * df['risk_flag'].mean():.1f} %")

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(make_risk_pie(df), use_container_width=True)
        with col_right:
            st.plotly_chart(make_offense_bar(df), use_container_width=True)

        st.plotly_chart(make_score_hist(df), use_container_width=True)

        if "entities_json" in df.columns:
            st.plotly_chart(make_top_persons_bar(df), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 – Email Explorer
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Search & Filter Emails")

        # ── Row 1: multi-keyword input + AND / OR toggle ───────────────────────
        kw_col, mode_col = st.columns([3, 1])
        keywords_raw = kw_col.text_input(
            "Keywords (comma-separated)",
            placeholder="e.g.  travel, massage, payment, Epstein",
            help=(
                "Enter one or more keywords separated by commas.  \n"
                "**AND** → email must contain *all* keywords.  \n"
                "**OR**  → email must contain *at least one* keyword."
            ),
        )
        match_mode = mode_col.radio(
            "Match mode",
            ["AND", "OR"],
            horizontal=True,
            help="AND = all keywords present · OR = any keyword present",
        )

        # ── Row 2: structured filters ──────────────────────────────────────────
        f1, f2 = st.columns(2)
        risk_filter    = f1.selectbox("Risk", ["All", "Problematic Only", "Safe Only"])
        offense_filter = f2.selectbox("Offense Category", ["All"] + OFFENSE_CATEGORIES)

        # ── Row 3: result count + search button ───────────────────────────────
        s1, s2 = st.columns([4, 1])
        max_results = s1.slider("Max results", min_value=5, max_value=100, value=20, step=5)
        search_btn  = s2.button("🔍  Search", use_container_width=True, type="primary")

        # Parse comma-separated keywords
        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

        # Show active keyword chips for clarity
        if keywords:
            chip_html = " ".join(
                f'<span style="background:#2E5FA3;color:#fff;padding:2px 8px;'
                f'border-radius:12px;font-size:0.8rem;margin:2px">{k}</span>'
                for k in keywords
            )
            st.markdown(
                f"**Active keywords ({match_mode}):** {chip_html}",
                unsafe_allow_html=True,
            )

        if search_btn or keywords:
            result_df, result_indices = keyword_search(
                df, keywords, match_mode, risk_filter, offense_filter, max_results
            )

            st.caption(f"Showing **{len(result_df)}** result(s)")
            st.dataframe(result_df, use_container_width=True, height=380)

            if result_indices:
                st.divider()
                st.subheader("Email Detail View")

                thread_options = [
                    str(df.loc[idx, "thread_id"]) for idx in result_indices
                ]
                selected_thread_id = st.selectbox(
                    "Select a thread ID to inspect", thread_options
                )

                if selected_thread_id:
                    matches = df[df["thread_id"].astype(str) == str(selected_thread_id)]
                    if not matches.empty:
                        with st.expander("📧 Full Email Details", expanded=True):
                            render_email_detail(matches.iloc[0])
        else:
            st.info(
                "Enter one or more keywords (comma-separated) or apply filters, "
                "then click **Search**."
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Semantic Search
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Semantic Similarity Search")
        st.caption(
            "Finds emails that are **conceptually related** to your query using "
            "FAISS + sentence embeddings — even if the exact words never appear.  \n"
            "The **Similarity** score is cosine similarity (0 → 1). "
            "Scores above ~0.4 indicate strong thematic relevance."
        )

        if not os.path.exists(EMBEDDINGS_PATH):
            st.warning(
                "Embeddings file not found.  "
                "Run `python main.py` to generate `data/embeddings.npy`."
            )
            st.stop()

        try:
            embedder, faiss_index = load_search_index()
        except Exception as e:
            st.error(f"Could not load semantic search index: {e}")
            st.stop()

        q1, q2 = st.columns([4, 1])
        query     = q1.text_input(
            "Natural-language query",
            placeholder="e.g.  young girls travel arrangements …",
        )
        k_results = q2.slider("Results", min_value=3, max_value=25, value=10)

        if st.button("🧠  Search Semantically", type="primary"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching …"):
                    sem_df = semantic_search(
                        query, df, embedder, faiss_index, k=k_results
                    )
                st.caption(f"Top **{len(sem_df)}** semantically similar threads")
                st.dataframe(sem_df, use_container_width=True, height=420)

                if not sem_df.empty:
                    st.divider()
                    st.subheader("Detail View")

                    thread_options = sem_df["Thread ID"].astype(str).tolist()

                    selected_thread_id = st.selectbox(
                        "Select a thread ID to inspect",
                        thread_options,
                        key="sem_thread_pick",
                    )

                    if selected_thread_id:
                        matches = df[df["thread_id"].astype(str) == str(selected_thread_id)]
                        if not matches.empty:
                            with st.expander("📧 Full Email Details", expanded=True):
                                render_email_detail(matches.iloc[0])
                        else:
                            st.info(
                                "Enter a natural-language query above and click **Search Semantically**."
                            )


if __name__ == "__main__":
    main()
