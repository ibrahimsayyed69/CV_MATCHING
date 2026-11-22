"""
Extract plain text from a CV file, regardless of format.
 
Designed so nothing gets silently dropped:
- PDFs: PyMuPDF reads the text layer PAGE BY PAGE; any page with (almost) no
  text but with an image on it is OCR'd, so a scanned page inside an otherwise
  normal PDF is not lost.
- DOCX: reads every paragraph, table cell, text box, header and footer.
- DOC (old Word format): read through Word itself (needs Word + pywin32).
- RTF: read with striprtf.
- Any CV whose extraction still looks poor is logged, and the index lists it
  as unreadable instead of ranking it on nothing.
"""
 
import os
import io
import logging
 
logger = logging.getLogger("cv_parser")
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    handler = logging.FileHandler("logs/extraction_log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
 
# Tesseract isn't on PATH on this machine — point pytesseract at it directly.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_DPI = 300
 
MIN_TEXT_LENGTH = 50   # a whole CV with less text than this counts as unreadable
MIN_PAGE_CHARS = 25    # a PDF page with less text than this (but an image on it) gets OCR'd
 
SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx", ".doc", ".rtf")
 
 
def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
 
    if ext == ".txt":
        return _extract_txt(path)
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".doc":
        return _extract_doc(path)
    if ext == ".rtf":
        return _extract_rtf(path)
 
    raise ValueError(f"Unsupported file type: {ext}")
 
 
def is_extraction_suspect(text: str) -> bool:
    """True when too little text came out — the CV can't be ranked reliably."""
    return len(text.strip()) < MIN_TEXT_LENGTH
 
 
def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
 
 
def _ocr_pixmap(pix) -> str:
    import pytesseract
    from PIL import Image
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)
 
 
def _extract_pdf(path: str) -> str:
    import pymupdf
 
    pages, ocr_pages = [], 0
    doc = pymupdf.open(path)
    try:
        for page in doc:
            blocks = page.get_text("blocks")
            text = "\n".join([b[4] for b in blocks if b[4].strip()])
            if len(text.strip()) < MIN_PAGE_CHARS and page.get_images(full=True):
                text = _ocr_pixmap(page.get_pixmap(dpi=OCR_DPI))
                ocr_pages += 1
            pages.append(text)
    finally:
        doc.close()
 
    text = "\n".join(pages)
    if ocr_pages:
        logger.info(f"'{path}': OCR used on {ocr_pages} page(s)")
    if is_extraction_suspect(text):
        logger.error(f"'{path}': only {len(text.strip())} chars extracted from PDF — REVIEW MANUALLY")
    return text
 
 
def _extract_docx(path: str) -> str:
    import docx
    from docx.oxml.ns import qn
 
    doc = docx.Document(path)
    W_P, W_T = qn("w:p"), qn("w:t")
    MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
 
    lines = []
    # Walking every <w:p> in the body covers normal paragraphs, table cells AND
    # text boxes (side-bar "Skills" boxes in CV templates are usually text boxes,
    # which python-docx's doc.paragraphs never returns).
    for p in doc.element.body.iter(W_P):
        if any(a.tag == MC_FALLBACK for a in p.iterancestors()):
            continue  # Word stores text boxes twice; skip the fallback copy
        text = "".join(
            t.text or "" for t in p.iter(W_T) if next(t.iterancestors(W_P), None) is p
        )
        if text.strip():
            lines.append(text.strip())
 
    seen = set()  # headers/footers often hold contact details
    for section in doc.sections:
        for part in (section.header, section.footer):
            for para in part.paragraphs:
                t = para.text.strip()
                if t and t not in seen:
                    seen.add(t)
                    lines.append(t)
 
    text = "\n".join(lines)
    if is_extraction_suspect(text):
        logger.error(f"'{path}': only {len(text.strip())} chars extracted from docx — REVIEW MANUALLY")
    return text
 
 
def _extract_doc(path: str) -> str:
    """Old .doc: let Microsoft Word open it and hand back the text."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        logger.error(f"'{path}': .doc needs 'pip install pywin32' and Microsoft Word — REVIEW MANUALLY")
        return ""
 
    pythoncom.CoInitialize()  # Streamlit runs scripts in a worker thread
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(os.path.abspath(path), ReadOnly=True, AddToRecentFiles=False)
        text = doc.Content.Text
        doc.Close(False)
    except Exception as e:
        logger.error(f"'{path}': Word could not open .doc ({e}) — REVIEW MANUALLY")
        return ""
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()
 
    if is_extraction_suspect(text):
        logger.error(f"'{path}': only {len(text.strip())} chars extracted from .doc — REVIEW MANUALLY")
    return text
 
 
def _extract_rtf(path: str) -> str:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        logger.error(f"'{path}': .rtf needs 'pip install striprtf' — REVIEW MANUALLY")
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = rtf_to_text(f.read())
    if is_extraction_suspect(text):
        logger.error(f"'{path}': only {len(text.strip())} chars extracted from .rtf — REVIEW MANUALLY")
    return text