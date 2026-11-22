TOP_N = 10
 
# --- CV REWRITE MODEL (Gemini via Google AI Studio) ---
REWRITE_MODEL = "gemini-3.5-flash"
 
# --- CV SOURCE ---
# Point straight at the folder where your CVs already live. It is only READ —
# nothing is copied, moved or changed. The r"" prefix keeps backslashes literal.
CV_SOURCE_TYPE = "local_folder"
LOCAL_CV_FOLDER = r"C:\Users\user\Pictures\Coolbuffs_DATA"
 
# Files whose NAME contains any of these (case-insensitive) are not treated as
# CVs: job descriptions, company profiles, etc. Skipped files are listed in
# logs/skipped_files.txt, so check it and add more patterns as you spot them.
EXCLUDE_NAME_CONTAINS = ["jd_", "job description", "corporate profile"]
 
# Folder names that mean the same thing. Applied AFTER lower-casing and after
# removing suffixes like " - SH" / " - SY". Add more as you notice them in the
# category dropdown.
CATEGORY_ALIASES = {
    "cyberdata analyst": "cyber data analyst",
}
 
# Results collapse near-identical CVs (same person filed under several role
# folders, or saved as both .pdf and .docx). Higher = merge only very close
# copies; lower = merge more aggressively. If the same person still shows up
# twice, try 0.95; if different people get merged, raise it.
DEDUP_THRESHOLD = 0.98