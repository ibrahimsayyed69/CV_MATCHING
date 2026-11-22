"""
Index all CVs OUTSIDE Streamlit — best for the big first run:

    python build_index.py

- Prints what was found / skipped, then live progress with a time estimate.
- Safe to stop (Ctrl+C) and re-run: progress is saved every 250 CVs and it
  carries on from there. Later runs only process new or changed CVs.
"""

import time

import config
from data_sources.local_folder import LocalFolderSource
from core import cache


def main():
    source = LocalFolderSource(config.LOCAL_CV_FOLDER)
    print(f"Found {len(source.list_cv_ids())} CV files; skipped {len(source.skipped)} "
          f"(reasons in logs/skipped_files.txt)")

    start = time.time()

    def on_progress(done, total):
        elapsed = time.time() - start
        left = elapsed / done * (total - done)
        print(f"\rIndexing {done}/{total}   ~{left / 60:.1f} min left   ", end="", flush=True)

    index = cache.get_index(source, on_progress, retry_unreadable=True)
    print()

    unreadable = [cid for cid, ok in zip(index.cv_ids, index.readable) if not ok]
    print(f"Done in {(time.time() - start) / 60:.1f} min. "
          f"{len(index.cv_ids) - len(unreadable)} CVs ready to rank, {len(unreadable)} unreadable.")
    if unreadable:
        print("Unreadable (review manually; details in logs/extraction_log.txt):")
        for cid in unreadable[:50]:
            print("  ", cid)
        if len(unreadable) > 50:
            print(f"   ... and {len(unreadable) - 50} more")


if __name__ == "__main__":
    main()