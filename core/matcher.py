"""
High-Precision Hybrid Matcher for CVs against Job Descriptions.

Combines four complementary signals:
1. Okapi BM25 Term Frequency & Length Normalization:
   Rewards deep, repeated technical expertise while normalizing for document length.
2. Exact Multi-Word Phrase & Bigram Matching:
   Differentiates real domain phrases (e.g., "machine learning", "data scientist",
   "power bi", "ci cd") from coincidental standalone word occurrences.
3. Headline & Role Title Alignment:
   Awards a relevance multiplier when target role titles appear in the candidate's
   primary resume header / current role title.
4. Semantic Vector Embeddings (all-MiniLM-L6-v2):
   Captures conceptual meaning, domain context, and synonym relationships.
5. AI Deep-Fit Analysis (Gemini):
   Performs senior technical recruiter evaluation with seniority verification,
   strengths, missing skill detection, and verdict reasoning.
"""

import re
import math
import json
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
from core.embedder import chunk_text, embed_chunks

# -----------------------------------------------------------------------------
# COMPREHENSIVE STOPWORDS LIST (Prevents noise words from hijacking IDF)
# -----------------------------------------------------------------------------
STOPWORDS = {
    # Common English words
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
    # Resume & Job Description generic boilerplate
    "year", "years", "experience", "looking", "role", "work", "candidate", "responsibilities",
    "requirements", "required", "skills", "job", "description", "company", "team", "ability",
    "strong", "etc", "including", "opportunity", "plus", "well", "must", "proficient", "knowledge",
    "working", "proven", "demonstrated", "familiarity", "good", "excellent", "environment", "seeking",
    "track", "record", "qualifications", "responsibilities", "ensure", "participate", "understanding",
    "provide", "perform", "closely", "ideal", "successful", "minimum", "maximum", "4+", "5+", "3+",
    "preferred", "duties", "across", "various", "multiple", "high", "hands", "day", "business",
}

_TOKEN_RE = re.compile(r"[a-z0-9+#]{2,}")


def extract_tokens(text: str) -> List[str]:
    """Extracts non-stopword tokens in lowercase."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def extract_bigrams(tokens: List[str]) -> Set[str]:
    """Extracts adjacent 2-token phrases (e.g. 'machine learning', 'data scientist')."""
    return {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}


# -----------------------------------------------------------------------------
# ADVANCED KEYWORD & PHRASE INDEX
# -----------------------------------------------------------------------------
class KeywordIndex:
    """
    Stores precomputed token frequencies, document lengths, and multi-word
    phrases across the entire candidate corpus for ultra-fast BM25 scoring.
    """

    def __init__(self, cv_ids: List[str], texts: Dict[str, str]):
        self.cv_ids = cv_ids
        self.texts = texts
        self.token_counts = []
        self.token_sets = []
        self.bigrams = []
        self.headers = []
        self.doc_lens = []

        for cid in cv_ids:
            txt = texts.get(cid, "")
            toks = extract_tokens(txt)
            self.doc_lens.append(len(toks))
            self.token_counts.append(Counter(toks))
            self.token_sets.append(set(toks))
            self.bigrams.append(extract_bigrams(toks))
            # Header: first ~600 chars (title, headline, contact, recent role)
            self.headers.append(set(extract_tokens(txt[:600])))

        # Document frequency and Okapi BM25 IDF
        df = Counter()
        for s in self.token_sets:
            df.update(s)

        n = max(len(cv_ids), 1)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
        self.avgdl = sum(self.doc_lens) / max(n, 1)

    def __getitem__(self, item):
        # Backwards compatibility: tuple unpack (token_sets, idf)
        if item == 0:
            return self.token_sets
        elif item == 1:
            return self.idf
        raise IndexError("KeywordIndex tuple indexing supports 0 (token_sets) and 1 (idf)")


def build_keyword_index(cv_ids: List[str], texts: Dict[str, str]) -> KeywordIndex:
    """Precomputes BM25 frequencies, phrases, and IDF weights across the corpus."""
    return KeywordIndex(cv_ids, texts)


# -----------------------------------------------------------------------------
# BM25 + PHRASE + TITLE SCORING
# -----------------------------------------------------------------------------
def _score_bm25_and_phrases(
    jd_text: str,
    kw_index: KeywordIndex,
    must_have_skills: Optional[List[str]] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> Tuple[np.ndarray, List[List[str]], List[List[str]], List[List[str]], np.ndarray]:
    """
    Computes Okapi BM25 scores, bigram phrase matches, header/title boosts,
    and skill coverage ratios for each candidate against the JD.
    """
    n = len(kw_index.cv_ids)
    scores = np.zeros(n, dtype=np.float32)
    matched_terms = [[] for _ in range(n)]
    matched_phrases = [[] for _ in range(n)]
    missing_terms = [[] for _ in range(n)]
    coverages = np.zeros(n, dtype=np.float32)

    jd_tokens = extract_tokens(jd_text)
    if not jd_tokens:
        return scores, matched_terms, matched_phrases, missing_terms, coverages

    jd_counts = Counter(jd_tokens)
    jd_bigrams = extract_bigrams(jd_tokens)

    # First line or first 120 chars as target title
    first_line = jd_text.strip().split("\n")[0]
    jd_title_tokens = set(extract_tokens(first_line))

    # Identify most discriminating technical keywords in the JD (IDF >= 0.8)
    key_terms = [t for t in jd_counts if kw_index.idf.get(t, 0) >= 0.8]
    if not key_terms:
        key_terms = list(jd_counts.keys())

    # Pre-parse must-have skills if provided
    must_haves_cleaned = [t.lower().strip() for t in (must_have_skills or []) if t.strip()]

    for i in range(n):
        tf_map = kw_index.token_counts[i]
        dl = kw_index.doc_lens[i]
        bm25 = 0.0

        # 1. Okapi BM25 with length normalization
        for t, q_tf in jd_counts.items():
            if t in tf_map:
                f = tf_map[t]
                w = kw_index.idf.get(t, 0)
                tf_comp = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / kw_index.avgdl)))
                bm25 += w * tf_comp

        # 2. Exact Bigram / Multi-word Phrase Bonus (+5.0 per matching phrase)
        shared_bigrams = sorted(list(jd_bigrams.intersection(kw_index.bigrams[i])))
        bm25 += 5.0 * len(shared_bigrams)

        # 3. Job Title & Header Alignment Bonus (+3.0 per matching title word)
        shared_title = jd_title_tokens.intersection(kw_index.headers[i])
        bm25 += 3.0 * len(shared_title)

        # 4. Core Skills Coverage Ratio
        shared_key = [t for t in key_terms if t in tf_map]
        missing_key = [t for t in key_terms if t not in tf_map]
        coverage = len(shared_key) / max(len(key_terms), 1)
        coverages[i] = coverage

        # Coverage penalty / bonus: Candidates missing most skills get penalized
        bm25 *= (0.45 + 0.55 * coverage)

        # 5. Must-Have Skills Enforcement (if specified)
        if must_haves_cleaned:
            candidate_raw = kw_index.texts.get(kw_index.cv_ids[i], "").lower()
            missing_must_haves = [mh for mh in must_haves_cleaned if mh not in candidate_raw]
            if missing_must_haves:
                bm25 *= 0.3  # Severe penalty if missing any mandatory skill

        scores[i] = bm25

        # Record matched & missing terms for candidate dossier
        matched_terms[i] = sorted(
            [t for t in jd_counts if t in tf_map],
            key=lambda t: -kw_index.idf.get(t, 0),
        )[:10]
        matched_phrases[i] = shared_bigrams[:6]
        missing_terms[i] = sorted(
            missing_key,
            key=lambda t: -kw_index.idf.get(t, 0),
        )[:8]

    return scores, matched_terms, matched_phrases, missing_terms, coverages


# -----------------------------------------------------------------------------
# MAIN RANKING FUNCTION
# -----------------------------------------------------------------------------
def rank_against_index(
    jd_text: str,
    index,
    top_n: int = 10,
    allowed=None,
    dedup_threshold: float = 0.98,
    keyword_index: Optional[KeywordIndex] = None,
    mode: str = "hybrid",
    must_have_skills: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Ranks candidates against a job description using hybrid precision matching.

    Modes:
      - 'hybrid': 65% BM25 & phrases, 35% semantic vectors (Recommended)
      - 'exact_skills': 80% BM25 & phrases, 20% semantic vectors
      - 'semantic': 70% semantic vectors, 30% BM25 & phrases
    """
    if not index.cv_ids or not jd_text.strip():
        return []

    # 1. Semantic Dense Embeddings (Chunk-level max-pool)
    jd_emb = embed_chunks(chunk_text(jd_text))
    sims = jd_emb @ index.E.T
    best_per_cv = np.maximum.reduceat(sims, index.starts, axis=1)
    sem_scores = best_per_cv.mean(axis=0)

    # 2. BM25, Exact Phrases, Title & Coverage Scoring
    if keyword_index is not None:
        if not isinstance(keyword_index, KeywordIndex):
            keyword_index = KeywordIndex(index.cv_ids, index.texts)

        (
            bm25_scores,
            all_matched_terms,
            all_matched_phrases,
            all_missing_terms,
            all_coverages,
        ) = _score_bm25_and_phrases(jd_text, keyword_index, must_have_skills=must_have_skills)

        # Min-max normalization for calibrated fusion
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-8:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        norm_bm25 = norm(bm25_scores)
        norm_sem = norm(sem_scores)

        # Select weights by mode
        if mode == "exact_skills":
            w_bm25, w_sem = 0.80, 0.20
        elif mode == "semantic":
            w_bm25, w_sem = 0.30, 0.70
        else:  # hybrid (default)
            w_bm25, w_sem = 0.65, 0.35

        # Factor in core skills coverage so candidates with all key requirements lead
        final_scores = (w_bm25 * norm_bm25 + w_sem * norm_sem) * (0.35 + 0.65 * all_coverages)
    else:
        final_scores = sem_scores
        all_matched_terms = [[] for _ in index.cv_ids]
        all_matched_phrases = [[] for _ in index.cv_ids]
        all_missing_terms = [[] for _ in index.cv_ids]
        all_coverages = np.zeros(len(index.cv_ids))

    # Mask unreadable & disallowed CVs
    final_scores = np.where(index.readable, final_scores, -np.inf)
    if allowed is not None:
        final_scores = np.where(allowed, final_scores, -np.inf)

    # 3. Deduplication and Selection
    selected = []
    sel_vecs = np.empty((0, index.doc_vecs.shape[1]), dtype=np.float32)
    for i in np.argsort(-final_scores):
        if final_scores[i] == -np.inf:
            break
        if len(selected) and float((sel_vecs @ index.doc_vecs[i]).max()) >= dedup_threshold:
            continue
        selected.append(int(i))
        sel_vecs = np.vstack([sel_vecs, index.doc_vecs[i]])
        if len(selected) == top_n:
            break

    # 4. Assemble Results
    results = []
    for i in selected:
        twins = np.where((index.doc_vecs @ index.doc_vecs[i] >= dedup_threshold) & index.readable)[0]
        results.append({
            "cv_id": index.cv_ids[i],
            "score": float(final_scores[i]),
            "semantic_score": float(sem_scores[i]),
            "skills_coverage": float(all_coverages[i]) if keyword_index is not None else 0.0,
            "also_in": [index.cv_ids[j] for j in twins if j != i],
            "matched_terms": all_matched_terms[i],
            "matched_phrases": all_matched_phrases[i],
            "missing_terms": all_missing_terms[i],
        })
    return results



