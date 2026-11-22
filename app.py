import os
import numpy as np
import pandas as pd
import streamlit as st

import config
from data_sources.local_folder import LocalFolderSource
from core import cache as cv_cache
from core.matcher import rank_against_index, build_keyword_index
from core.rewriter import tailor_cv, export_resume_to_pdf

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentSync | Semantic CV Matcher & ATS Tailor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM STYLING (Editorial Theme: Warm Paper, Terracotta, Forest Green)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root application background & base typography */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FAF7F2 !important;
        color: #1A1A1E !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Universal typography */
    h2, h3, h4, h5, h6, p, span, label, div {
        color: #1A1A1E;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar: #FFFFFF with 1px border #E7E0D5 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E7E0D5 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1A1A1E;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p[data-testid="stCaptionContainer"] {
        color: #6B6660 !important;
    }

    /* Cards & Containers: #FFFFFF with 1px border #E7E0D5 */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E7E0D5 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(26, 26, 30, 0.02) !important;
    }

    /* Secondary / Muted Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #6B6660 !important;
        font-size: 0.85rem !important;
    }

    /* Hero Banner: #FFFFFF with 1px border #E7E0D5 */
    .hero-banner {
        padding: 1.5rem 1.9rem;
        background-color: #FFFFFF;
        border: 1px solid #E7E0D5;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 4px rgba(26, 26, 30, 0.03);
    }

    /* Pill Badge */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #C2410C;
        color: #FFFFFF !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 0.55rem;
    }

    /* Main Title ONLY: Fraunces Serif */
    .hero-title, h1.hero-title {
        font-family: 'Fraunces', 'Source Serif 4', Georgia, serif !important;
        font-size: 2.45rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        color: #1A1A1E !important;
        line-height: 1.16 !important;
        margin: 0;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem;
        color: #6B6660 !important;
        margin-top: 0.45rem;
        margin-bottom: 0;
        max-width: 960px;
        line-height: 1.5;
    }

    /* Primary Buttons: #C2410C (Burnt Orange) / Hover #9A3412 */
    div.stButton > button, div.stButton > button[kind="primary"] {
        background-color: #C2410C !important;
        color: #FFFFFF !important;
        border: 1px solid #C2410C !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    div.stButton > button:hover, div.stButton > button[kind="primary"]:hover {
        background-color: #9A3412 !important;
        border-color: #9A3412 !important;
        color: #FFFFFF !important;
    }

    /* Secondary Buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #1A1A1E !important;
        border: 1px solid #E7E0D5 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #EFE9DC !important;
        border-color: #E7E0D5 !important;
        color: #1A1A1E !important;
    }

    /* Accent Tags / Chips: #EFE9DC (Beige chip) */
    .tag-chip {
        display: inline-block;
        background-color: #EFE9DC !important;
        color: #1A1A1E !important;
        border: 1px solid #E0D8C8 !important;
        border-radius: 9999px;
        padding: 3px 11px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 3px 0;
    }

    .phrase-chip {
        display: inline-block;
        background-color: #EFE9DC !important;
        color: #C2410C !important;
        border: 1px solid #E0D8C8 !important;
        border-radius: 9999px;
        padding: 3px 11px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 3px 0;
    }

    /* Warning / Missing / Unreadable: #92400E on #FEF3C7 */
    .missing-chip {
        display: inline-block;
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FDE68A !important;
        border-radius: 9999px;
        padding: 3px 11px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 3px 0;
    }

    .duplicate-notice {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FDE68A !important;
        border-left: 4px solid #C2410C !important;
        padding: 9px 14px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    div[data-testid="stAlert"] {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FDE68A !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] * {
        color: #92400E !important;
    }

    /* Match Score Badges: Success #1B5E3B (Deep Forest Green) */
    .badge-score-high {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: #EAF5EE !important;
        color: #1B5E3B !important;
        border: 1px solid #A8D8BF !important;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 5px 14px;
        border-radius: 9999px;
    }
    .badge-score-mid {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: #EFE9DC !important;
        color: #1A1A1E !important;
        border: 1px solid #E0D8C8 !important;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 5px 14px;
        border-radius: 9999px;
    }
    .badge-score-fair {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FDE68A !important;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 5px 14px;
        border-radius: 9999px;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3.5rem 1.8rem;
        border: 1px dashed #E7E0D5;
        border-radius: 14px;
        background-color: #FFFFFF;
    }
    .empty-state-icon {
        font-size: 2.8rem;
        margin-bottom: 0.75rem;
    }
    .empty-state-title {
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.4rem;
        color: #1A1A1E;
    }
    .empty-state-desc {
        color: #6B6660;
        font-size: 0.92rem;
        max-width: 440px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Text areas, inputs & dropdowns */
    .stTextArea textarea, .stTextInput input, [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E7E0D5 !important;
        color: #1A1A1E !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #C2410C !important;
        box-shadow: 0 0 0 1px #C2410C !important;
    }

    /* Multiselect beige tags */
    [data-baseweb="tag"] {
        background-color: #EFE9DC !important;
        color: #1A1A1E !important;
        border: 1px solid #E0D8C8 !important;
    }
    [data-baseweb="tag"] * {
        color: #1A1A1E !important;
    }

    /* Progress bar in terracotta */
    .stProgress > div > div > div > div {
        background-color: #C2410C !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #E7E0D5 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #C2410C !important;
        border-bottom: 2px solid #C2410C !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6B6660 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1A1A1E !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6B6660 !important;
        font-weight: 600 !important;
    }

    /* Dataframe / Table */
    [data-testid="stDataFrame"] {
        border: 1px solid #E7E0D5 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SOURCE & SESSION STATE
# -----------------------------------------------------------------------------
@st.cache_resource
def get_source():
    if config.CV_SOURCE_TYPE == "local_folder":
        return LocalFolderSource(config.LOCAL_CV_FOLDER)
    raise ValueError(f"Unknown CV_SOURCE_TYPE: {config.CV_SOURCE_TYPE}")

source = get_source()

for key in ("ranked", "index", "tailored_cv", "jd_at_search", "kw_index", "kw_sig", "selected_cand_idx"):
    st.session_state.setdefault(key, None)

if "jd_text_area" not in st.session_state:
    st.session_state["jd_text_area"] = ""

def render_score_badge(score: float) -> str:
    pct = round(score * 100, 1)
    if pct >= 70:
        return f'<span class="badge-score-high">🔥 {pct}% High Match</span>'
    elif pct >= 50:
        return f'<span class="badge-score-mid">⚡ {pct}% Solid Match</span>'
    else:
        return f'<span class="badge-score-fair">🎯 {pct}% Fair Match</span>'

# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">✦ AI Talent Discovery</div>
        <h1 class="hero-title">TalentSync | Semantic CV Matcher</h1>
        <p class="hero-subtitle">
            Surfaces top candidate resumes using multi-signal Okapi BM25, exact multi-word phrase matching, 
            seniority alignment, and neural embeddings — with one-click AI Deep-Fit Evaluation and ATS tailoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

num_cvs = len(source.list_cv_ids())
counts = source.category_counts()
num_categories = len(counts)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Candidate Repository")
    st.caption(f"Folder: `{os.path.basename(config.LOCAL_CV_FOLDER)}`")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Total CVs", f"{num_cvs:,}")
    with col_sb2:
        st.metric("Roles", f"{num_categories}")
    
    st.divider()
    st.markdown("#### ⚡ Maintenance Actions")
    if st.button("🔄 Rescan CV Folder", use_container_width=True, help="Scan directory for newly added or modified CVs"):
        get_source.clear()
        st.session_state.ranked = None
        st.session_state.tailored_cv = None
        st.success("Cache cleared! Rescanning...")
        st.rerun()

    if st.button("🧹 Clear Results", use_container_width=True, help="Clear current search query and candidate list"):
        st.session_state.ranked = None
        st.session_state.tailored_cv = None
        st.session_state["jd_text_area"] = ""
        st.rerun()

    # Skipped files inspection
    if source.skipped:
        st.divider()
        with st.expander(f"⚠️ {len(source.skipped)} Skipped Files", expanded=False):
            st.caption("Office lock files, shortcuts, or excluded name patterns:")
            filter_reason = st.selectbox(
                "Filter by skip reason:",
                ["All"] + sorted(list({reason for _, reason in source.skipped}))
            )
            displayed_skipped = [
                (p, r) for p, r in source.skipped
                if filter_reason == "All" or r == filter_reason
            ]
            for path, reason in displayed_skipped[:80]:
                st.markdown(f"**`{os.path.basename(path)}`**  \n<span style='font-size:0.78rem; color:#92400E;'>↳ {reason}</span>", unsafe_allow_html=True)
            if len(displayed_skipped) > 80:
                st.caption(f"Showing first 80 of {len(displayed_skipped)}. Full list in `logs/skipped_files.txt`.")

    st.divider()
    with st.expander("ℹ️ How Precision Matching Works"):
        st.markdown(
            """
            - **Okapi BM25 & Stopwords:** Completely eliminates noise words ('we', 'looking', 'experience') and rewards depth of technical usage.
            - **Exact Phrases & Bigrams:** Distinguishes multi-word concepts (e.g. 'machine learning', 'data scientist') from isolated words.
            - **Seniority & Title Alignment:** Recognizes candidate headlines and role seniority.
            - **Coverage Multiplier:** Candidates covering 90%+ of required skills outrank partial matches.
            """
        )

# -----------------------------------------------------------------------------
# MAIN TWO-COLUMN WORKFLOW
# -----------------------------------------------------------------------------
col_search, col_results = st.columns([5, 6], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: JOB CRITERIA & SEARCH SETTINGS
# -----------------------------------------------------------------------------
with col_search:
    with st.container(border=True):
        st.markdown("### 📋 1. Target Job Description")
        st.caption("Paste the requirements, responsibilities, and desired technical skills below.")

        jd_text = st.text_area(
            "Job Description Content",
            key="jd_text_area",
            height=260,
            placeholder="Paste the target job description here...",
            label_visibility="collapsed",
        )

        words = len(jd_text.split()) if jd_text else 0
        chars = len(jd_text) if jd_text else 0
        st.caption(f"📝 Word count: **{words}** words ({chars} characters)")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### 🏷️ 2. Target Category Filtering")
        st.caption("Restrict search to specific role folders (leave blank to search all categories).")

        chosen_categories = st.multiselect(
            "Filter by category",
            sorted(counts.keys()),
            format_func=lambda c: f"{c.title()} ({counts[c]} CVs)",
            placeholder="Select one or more categories (default: all)...",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        search_clicked = st.button("🚀 Search & Rank Top Candidates", type="primary", use_container_width=True)

    # Search Execution
    if search_clicked:
        if not jd_text.strip():
            st.warning("⚠️ Please paste a job description before searching.")
        else:
            progress_bar = st.progress(0.0, text="Checking candidate index...")

            def on_progress(done, total):
                progress_bar.progress(done / total, text=f"Indexing CV files: {done}/{total}")

            index = cv_cache.get_index(source, on_progress)
            progress_bar.empty()

            # Rebuild keyword index if CV set changed
            sig = (len(index.cv_ids), index.E.shape[0], sum(len(t) for t in index.texts.values()))
            if st.session_state.kw_sig != sig or st.session_state.kw_index is None:
                with st.spinner("Building high-precision BM25 and phrase index..."):
                    st.session_state.kw_index = build_keyword_index(index.cv_ids, index.texts)
                st.session_state.kw_sig = sig

            # Filter categories if chosen
            allowed = None
            if chosen_categories:
                wanted = set(chosen_categories)
                allowed = np.array([source.get_category(cid) in wanted for cid in index.cv_ids])

            with st.spinner("⚡ Running high-precision ranking..."):
                ranked_candidates = rank_against_index(
                    jd_text,
                    index,
                    top_n=config.TOP_N,
                    allowed=allowed,
                    dedup_threshold=config.DEDUP_THRESHOLD,
                    keyword_index=st.session_state.kw_index,
                    mode="hybrid",
                )

            st.session_state.ranked = ranked_candidates
            st.session_state.index = index
            st.session_state.jd_at_search = jd_text
            st.session_state.tailored_cv = None
            st.session_state.selected_cand_idx = 0
            st.rerun()

# -----------------------------------------------------------------------------
# RIGHT COLUMN: CANDIDATE DOSSIER & RESULTS
# -----------------------------------------------------------------------------
with col_results:
    ranked = st.session_state.ranked
    index = st.session_state.index

    # Check for unreadable CVs
    if index is not None:
        unreadable = [cid for cid, ok in zip(index.cv_ids, index.readable) if not ok]
        if unreadable:
            st.warning(f"⚠️ {len(unreadable)} CV(s) could not be parsed and were excluded from ranking.")
            with st.expander("Show Unreadable CVs"):
                for cid in unreadable:
                    st.text(cid)

    # EMPTY STATE
    if not ranked:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">🎯</div>
                <div class="empty-state-title">No Search Results Yet</div>
                <div class="empty-state-desc">
                    Paste a job description on the left, optionally pick role categories, 
                    and click <b>Search & Rank Top Candidates</b> to surface matching profiles.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Results Header with quick metrics
        top_score_pct = round(ranked[0]["score"] * 100, 1)
        avg_score_pct = round(np.mean([r["score"] for r in ranked[:5]]) * 100, 1) if ranked else 0

        st.markdown(f"### 🏆 Top {len(ranked)} Matching Candidates")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Total Matches", f"{len(ranked)} CVs")
        with col_res2:
            st.metric("Top Fit Score", f"{top_score_pct}%")
        with col_res3:
            st.metric("Top-5 Avg", f"{avg_score_pct}%")

        # Quick Overview Table (collapsible)
        with st.expander("📊 View All Candidates Comparison Table", expanded=False):
            summary_data = []
            for rank_num, r in enumerate(ranked, 1):
                meta = source.get_cv_metadata(r["cv_id"])
                cov = f"{round(r.get('skills_coverage', 0) * 100)}%"
                summary_data.append({
                    "Rank": f"#{rank_num}",
                    "Candidate Name": meta["name"].title(),
                    "Category": meta["category"].title(),
                    "Overall Fit": f"{round(r['score'] * 100, 1)}%",
                    "Skills Coverage": cov,
                    "Key Phrases": ", ".join(r.get("matched_phrases", [])[:3]),
                    "Top Keywords": ", ".join(r.get("matched_terms", [])[:4]),
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Candidate Selector Options
        candidate_options = []
        for i, r in enumerate(ranked):
            meta = source.get_cv_metadata(r["cv_id"])
            pct = round(r["score"] * 100, 1)
            name = meta["name"].title()
            cat = meta["category"].title()
            duplicate_tag = f" (+{len(r['also_in'])} duplicates)" if r["also_in"] else ""
            candidate_options.append(f"#{i+1} • {name}  —  {cat}  —  {pct}% fit{duplicate_tag}")

        cur_idx = st.session_state.selected_cand_idx or 0
        if cur_idx >= len(ranked):
            cur_idx = 0

        nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
        with nav_col1:
            if st.button("◀ Prev", disabled=(cur_idx == 0), use_container_width=True):
                st.session_state.selected_cand_idx = max(0, cur_idx - 1)
                st.session_state.tailored_cv = None
                st.rerun()
        with nav_col2:
            selected_idx = st.selectbox(
                "Select Candidate:",
                options=list(range(len(ranked))),
                index=cur_idx,
                format_func=lambda i: candidate_options[i],
                label_visibility="collapsed",
            )
            if selected_idx != cur_idx:
                st.session_state.selected_cand_idx = selected_idx
                st.session_state.tailored_cv = None
                st.rerun()
        with nav_col3:
            if st.button("Next ▶", disabled=(cur_idx == len(ranked) - 1), use_container_width=True):
                st.session_state.selected_cand_idx = min(len(ranked) - 1, cur_idx + 1)
                st.session_state.tailored_cv = None
                st.rerun()

        selected_candidate = ranked[selected_idx]
        selected_cid = selected_candidate["cv_id"]
        cand_meta = source.get_cv_metadata(selected_cid)
        cand_score = selected_candidate["score"]
        cand_pct = round(cand_score * 100, 1)
        cand_cov = selected_candidate.get("skills_coverage", 0.0)
        cand_cov_pct = round(cand_cov * 100, 1)

        # Candidate Dossier Card
        with st.container(border=True):
            header_col1, header_col2 = st.columns([3, 2])
            with header_col1:
                st.markdown(f"## {cand_meta['name'].title()}")
                st.markdown(f"**Primary Category:** `{cand_meta['category'].title()}`")
            with header_col2:
                st.markdown(
                    f"""
                    <div style="text-align: right; padding-top: 6px;">
                        {render_score_badge(cand_score)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Match Score Confidence Bar & Skills Coverage
            norm_score = max(0.0, min(1.0, cand_score))
            st.progress(norm_score, text=f"Overall Match Fit: {cand_pct}%  •  Technical Skills Coverage: {cand_cov_pct}%")

            # Matched Phrases
            matched_phrases = selected_candidate.get("matched_phrases", [])
            if matched_phrases:
                st.markdown("**🔗 Exact Multi-Word Phrases Found:**")
                phrase_chips = "".join([f'<span class="phrase-chip">"{p}"</span>' for p in matched_phrases])
                st.markdown(f"<div>{phrase_chips}</div>", unsafe_allow_html=True)

            # Rarest Matched Terms Chips
            matched_terms = selected_candidate.get("matched_terms", [])
            if matched_terms:
                st.markdown("**🎯 Key Technical Skills Matched:**")
                chips_html = "".join([f'<span class="tag-chip">#{term}</span>' for term in matched_terms])
                st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

            # Missing Important Skills
            missing_terms = selected_candidate.get("missing_terms", [])
            if missing_terms:
                st.markdown("**⚠️ Notable Job Requirements NOT Evidenced in CV:**")
                missing_chips = "".join([f'<span class="missing-chip">✗ {t}</span>' for t in missing_terms])
                st.markdown(f"<div>{missing_chips}</div>", unsafe_allow_html=True)

            # Duplicate Notice
            if selected_candidate.get("also_in"):
                st.markdown(
                    f"""
                    <div class="duplicate-notice">
                        📂 Also filed in {len(selected_candidate['also_in'])} other location(s) (automatically de-duplicated)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Tabs for Deep Dive
        tab_text, tab_tailor, tab_file = st.tabs([
            "📄 Extracted Resume",
            "✨ ATS Tailored Resume",
            "📁 File & Source Details",
        ])

        with tab_text:
            extracted_text = index.texts.get(selected_cid, "")
            st.caption(f"Raw extracted text preview ({len(extracted_text.split())} words):")
            st.text_area(
                "Resume Text Preview",
                value=extracted_text,
                height=350,
                label_visibility="collapsed",
            )

        with tab_tailor:
            st.caption("Tailor and optimize this resume to match the target job description using Gemini AI:")
            
            tailor_col1, tailor_col2, tailor_col3 = st.columns([2, 1, 1])
            with tailor_col1:
                tailor_clicked = st.button("✨ Tailor this CV with Gemini AI", type="primary", use_container_width=True)
            with tailor_col2:
                if st.session_state.tailored_cv:
                    pdf_bytes = export_resume_to_pdf(st.session_state.tailored_cv, candidate_name=cand_meta['name'])
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name=f"Tailored_{cand_meta['name'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            with tailor_col3:
                if st.session_state.tailored_cv:
                    st.download_button(
                        label="📄 Download TXT",
                        data=st.session_state.tailored_cv,
                        file_name=f"Tailored_{cand_meta['name'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            if tailor_clicked:
                with st.spinner("🤖 Rewriting & optimizing resume for ATS compliance (Gemini AI)..."):
                    try:
                        st.session_state.tailored_cv = tailor_cv(
                            extracted_text, st.session_state.jd_at_search
                        )
                        st.success("✅ Resume successfully tailored!")
                    except Exception as err:
                        st.error(f"Error tailoring CV: {err}")

            if st.session_state.tailored_cv:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.text_area(
                    "Tailored Output (ATS Optimized)",
                    value=st.session_state.tailored_cv,
                    height=380,
                )

        with tab_file:
            st.markdown("#### Primary File Path:")
            st.code(cand_meta["source_ref"], language="text")
            
            if selected_candidate.get("also_in"):
                st.markdown("#### Duplicate Copies in Repository:")
                for other_cid in selected_candidate["also_in"]:
                    st.code(source.get_cv_metadata(other_cid)["source_ref"], language="text")
