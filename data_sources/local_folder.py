"""
Reads CVs from a local folder tree (.pdf .docx .doc .rtf .txt).
 
- Scans every subfolder, however deep.
- Category = the folder the file sits in (lower-cased, with status suffixes such
  as " - SH" / " - SY" removed). A CV sitting directly inside a top-level
  batch folder has no role folder, so it is "uncategorized".
- Anything it can't use (Office lock files, shortcuts, csv, excluded names...)
  is not ingested silently: it is listed in logs/skipped_files.txt.
"""
 
import os
import re
from collections import Counter
from typing import List, Dict
 
import config
from data_sources.base import CVSource
from core.parser import extract_text, SUPPORTED_EXTENSIONS
 
SKIP_REPORT = os.path.join("logs", "skipped_files.txt")
 
 
class LocalFolderSource(CVSource):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self._files = {}        # cv_id -> full path
        self._categories = {}   # cv_id -> category
        self.skipped = []       # [(full_path, reason)]
 
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"CV folder not found: {folder_path}  (check LOCAL_CV_FOLDER in config.py)")
 
        for root, _dirs, filenames in os.walk(folder_path):
            for fname in filenames:
                full = os.path.join(root, fname)
                reason = self._skip_reason(fname)
                if reason:
                    self.skipped.append((full, reason))
                    continue
                rel_parts = os.path.relpath(full, folder_path).split(os.sep)
                cv_id = "/".join(rel_parts)  # unique, keeps the extension (a.pdf vs a.docx)
                self._files[cv_id] = full
                self._categories[cv_id] = self._category_from(rel_parts)
 
        self._write_skip_report()
 
    # ---- what to ingest -------------------------------------------------
    @staticmethod
    def _skip_reason(fname: str):
        low = fname.lower()
        if low.startswith("~$"):
            return "Office temporary/lock file"
        if low.startswith("."):
            return "hidden file"
        ext = os.path.splitext(low)[1]
        if ext not in SUPPORTED_EXTENSIONS:
            return f"unsupported file type ({ext or 'no extension'})"
        for pattern in config.EXCLUDE_NAME_CONTAINS:
            if pattern.lower() in low:
                return f"excluded by name pattern '{pattern}'"
        return None
 
    @staticmethod
    def _category_from(rel_parts: List[str]) -> str:
        folders = rel_parts[:-1]
        if len(folders) <= 1:      # directly in the root or in a top-level batch folder
            return "uncategorized"
        name = folders[-1].strip().lower()
        name = re.sub(r"\s*[-–]\s*(sh|sy|s|e|fn)\s*$", "", name)   # " - SH", "-SY", "– FN" ...
        name = re.sub(r"\s+", " ", name).strip()
        return config.CATEGORY_ALIASES.get(name, name) or "uncategorized"
 
    def _write_skip_report(self):
        os.makedirs("logs", exist_ok=True)
        counts = Counter(reason for _p, reason in self.skipped)
        with open(SKIP_REPORT, "w", encoding="utf-8") as f:
            f.write(f"{len(self._files)} CV files ingested, {len(self.skipped)} files skipped\n\n")
            for reason, n in counts.most_common():
                f.write(f"{n:6d}  {reason}\n")
            f.write("\n")
            for path, reason in sorted(self.skipped, key=lambda x: (x[1], x[0])):
                f.write(f"[{reason}] {path}\n")
 
    # ---- CVSource interface --------------------------------------------
    def list_cv_ids(self) -> List[str]:
        return list(self._files.keys())
 
    def get_cv_text(self, cv_id: str) -> str:
        return extract_text(self._files[cv_id])
 
    def get_cv_version(self, cv_id: str) -> str:
        st = os.stat(self._files[cv_id])
        return f"{st.st_mtime_ns}-{st.st_size}"
 
    def get_cv_metadata(self, cv_id: str) -> Dict:
        stem = os.path.splitext(cv_id.split("/")[-1])[0]
        return {
            "name": stem.replace("_", " ").strip(),
            "category": self._categories[cv_id],
            "source_ref": self._files[cv_id],
        }
 
    # ---- extras used by the UI -------------------------------------------
    def get_category(self, cv_id: str) -> str:
        return self._categories[cv_id]
 
    def category_counts(self) -> Dict[str, int]:
        return dict(Counter(self._categories.values()))