import os
import sys
from core.parser import extract_text, is_extraction_suspect, SUPPORTED_EXTENSIONS

def process_single_cv(file_path: str):
    text = extract_text(file_path)
    if is_extraction_suspect(text):
        print(f"[WARNING] Suspect extraction for: {file_path}")
    else:
        print(f"[SUCCESS] Extracted {len(text)} characters from {os.path.basename(file_path)}")
        print("--- Preview (First 300 chars) ---")
        print(text[:300])

def process_cv_folder(folder_path: str):
    total, failed = 0, 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                total += 1
                full_path = os.path.join(root, file)
                try:
                    text = extract_text(full_path)
                    if is_extraction_suspect(text):
                        failed += 1
                except Exception as e:
                    print(f"[ERROR] {file}: {e}")
                    failed += 1
    print(f"\nDone! Processed {total} CVs. Suspect/Failed: {failed} (check logs/extraction_log.txt)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_parser.py <path_to_cv_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isfile(target):
        process_single_cv(target)
    elif os.path.isdir(target):
        process_cv_folder(target)
    else:
        print(f"Path not found: {target}")