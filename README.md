# CV Matcher & ATS Tailor

Paste a job description -> get top 10 matching candidates from the CV
database -> pick one -> get their CV rewritten to be ATS-friendly for
that job.

## Setup

```
pip install -r requirements.txt
```

**OCR needs one non-pip install:** scanned/image-based PDFs are handled via
Tesseract OCR, which is a separate program, not a Python package.

Windows: download and run the installer from
https://github.com/UB-Mannheim/tesseract/wiki — after installing, if
`pytesseract` can't find it automatically, add this near the top of
`core/parser.py`:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Set your Anthropic API key (needed only for the CV-rewrite step):

Windows PowerShell:
```
$env:ANTHROPIC_API_KEY="your-key-here"
```

## Run

```
streamlit run app.py
```

It opens in your browser. It's currently pointed at 5 sample CVs in
`sample_data/cvs/` so you can test the whole flow today.

## Swapping in the real database (when you get access)

1. Open `data_sources/base.py` — this is the contract every source follows.
2. Create a new file, e.g. `data_sources/sql_source.py`, with a class that
   implements `list_cv_ids()`, `get_cv_text()`, `get_cv_metadata()` against
   your real database/API.
3. In `config.py`, change `CV_SOURCE_TYPE` and point `app.py`'s
   `get_source()` at your new class.

Nothing in `core/matcher.py`, `core/rewriter.py`, or `app.py` needs to
change — they only talk to the `CVSource` interface, not to files directly.

## Extraction robustness

- PDFs: PyMuPDF reads the text layer first; if it comes back too short
  (< 50 chars — usually means a scanned/image-based CV), it automatically
  re-tries via OCR on each page.
- DOCX: reads both paragraphs and tables (skills/education are often in a
  table, which a paragraphs-only reader misses).
- Any CV where extraction still looks too short even after OCR is logged to
  `logs/extraction_log.txt` AND flagged with ⚠️ in the UI if it shows up in
  a ranked result — so a badly-parsed CV never silently disappears, it gets
  surfaced for manual review instead.

## Notes

- Matching uses `sentence-transformers` (all-MiniLM-L6-v2) — local, free,
  semantic (not just exact keyword match). First run downloads the model
  (~90MB), needs internet once.
- Rewriting uses Claude (`claude-sonnet-4-6`) via the Anthropic API — this
  step needs internet + your API key every time it runs.
- Currently outputs tailored CVs as plain text. If you want a formatted
  .docx download instead, say so and I'll add it.

##
current status of project and app code
import os
import numpy as np
import streamlit as st

import config
from data_sources.local_folder import LocalFolderSource
from core import cache as cv_cache
from core.matcher import rank_against_index, build_keyword_index
from core.rewriter import tailor_cv

st.set_page_config(page_title="CV Matcher", layout="wide")
st.title("CV Matcher & ATS Tailor")


@st.cache_resource
def get_source():
    if config.CV_SOURCE_TYPE == "local_folder":
        return LocalFolderSource(config.LOCAL_CV_FOLDER)
    raise ValueError(f"Unknown CV_SOURCE_TYPE: {config.CV_SOURCE_TYPE}")


source = get_source()

for key in ("ranked", "index", "tailored_cv", "jd_at_search", "kw_index", "kw_sig"):
    st.session_state.setdefault(key, None)

# ---- sidebar: what was found / skipped -----------------------------------
st.sidebar.write(f"**{len(source.list_cv_ids())}** CV files found")
if st.sidebar.button("Rescan CV folder"):
    get_source.clear()
    st.rerun()
if source.skipped:
    with st.sidebar.expander(f"{len(source.skipped)} files skipped"):
        st.caption("Full list: logs/skipped_files.txt")
        for path, reason in source.skipped[:200]:
            st.text(f"{os.path.basename(path)}\n  → {reason}")

# ---- search inputs ----------------------------------------------------------
jd_text = st.text_area("Paste the job description", height=200)

counts = source.category_counts()
chosen = st.multiselect(
    "Limit to categories (optional — leave empty to search everything)",
    sorted(counts),
    format_func=lambda c: f"{c} ({counts[c]})",
)

if st.button("Search candidates", type="primary"):
    if not jd_text.strip():
        st.warning("Paste a job description first.")
    else:
        bar = st.progress(0.0, text="Checking for new or changed CVs...")

        def on_progress(done, total):
            bar.progress(done / total, text=f"Indexing new/changed CVs: {done}/{total}")

        index = cv_cache.get_index(source, on_progress)
        bar.empty()

        # Keyword index (rare-word weights): rebuilt only when the CV set changes.
        sig = (len(index.cv_ids), index.E.shape[0], sum(len(t) for t in index.texts.values()))
        if st.session_state.kw_sig != sig:
            with st.spinner("Preparing keyword index (only when CVs change)..."):
                st.session_state.kw_index = build_keyword_index(index.cv_ids, index.texts)
            st.session_state.kw_sig = sig

        allowed = None
        if chosen:
            wanted = set(chosen)
            allowed = np.array([source.get_category(cid) in wanted for cid in index.cv_ids])

        with st.spinner("Ranking candidates..."):
            st.session_state.ranked = rank_against_index(
                jd_text, index, top_n=config.TOP_N, allowed=allowed,
                dedup_threshold=config.DEDUP_THRESHOLD,
                keyword_index=st.session_state.kw_index,
            )
        st.session_state.index = index
        st.session_state.jd_at_search = jd_text
        st.session_state.tailored_cv = None

# ---- results ----------------------------------------------------------------
ranked, index = st.session_state.ranked, st.session_state.index

if index is not None:
    unreadable = [cid for cid, ok in zip(index.cv_ids, index.readable) if not ok]
    if unreadable:
        st.warning(f"{len(unreadable)} CV(s) could not be read and were NOT ranked — review them manually.")
        with st.expander("Show unreadable CVs"):
            for cid in unreadable:
                st.text(cid)

if ranked:
    st.subheader(f"Top {len(ranked)} candidates")

    labels = []
    for r in ranked:
        meta = source.get_cv_metadata(r["cv_id"])
        also = f"  ·  also filed under {len(r['also_in'])} other folder(s)" if r["also_in"] else ""
        labels.append(f'{meta["name"]}  —  {meta["category"]}  —  match {r["score"]:.2f}{also}')

    pick = st.radio("Select a candidate:", list(range(len(ranked))), format_func=lambda i: labels[i])
    selected = ranked[pick]
    selected_id = selected["cv_id"]

    with st.expander("Where this CV is filed"):
        st.text(source.get_cv_metadata(selected_id)["source_ref"])
        for other in selected["also_in"]:
            st.text(source.get_cv_metadata(other)["source_ref"])

    if selected.get("matched_terms"):
        st.caption("Rarest job-description terms found in this CV: " + ", ".join(selected["matched_terms"]))

    with st.expander("Preview the extracted text (check it was read properly)"):
        st.text(index.texts[selected_id][:4000])

    if st.button("Tailor this CV for the job description"):
        with st.spinner("Rewriting CV..."):
            st.session_state.tailored_cv = tailor_cv(
                index.texts[selected_id], st.session_state.jd_at_search
            )
elif ranked is not None:
    st.info("No readable CVs matched. Check the category filter and logs/skipped_files.txt.")

if st.session_state.tailored_cv:
    st.subheader("ATS-tailored CV")
    st.text_area("Result", st.session_state.tailored_cv, height=400)
    st.download_button("Download as .txt", st.session_state.tailored_cv, file_name="tailored_cv.txt")


##