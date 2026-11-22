"""
Rewrites a candidate's CV to better match a job description and pass
ATS (Applicant Tracking System) parsing.

Uses the Gemini API via a Google AI Studio key.
Requires the GEMINI_API_KEY environment variable.
Get a free key at https://aistudio.google.com/apikey
"""

import os
from google import genai
from dotenv import load_dotenv
import config

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set the GEMINI_API_KEY environment variable before using the rewriter."
            )
        _client = genai.Client(api_key=api_key)
    return _client


ATS_REWRITE_PROMPT = """You are an expert résumé writer specializing in ATS \
(Applicant Tracking System) optimization.

Rewrite the candidate's CV below so it is tailored to the target job \
description, while staying strictly truthful to the candidate's actual \
experience — do not invent skills, employers, dates, or achievements.

ATS-friendly rules to follow:
- Plain section headings (e.g. "Experience", "Skills", "Education") — no tables, columns, icons, or text boxes
- Use standard fonts/formatting implied by plain text (no special characters for bullets other than "-")
- Naturally incorporate exact keywords and phrasing from the job description where the candidate's real experience supports it
- Quantify achievements where the original CV already implies numbers
- Keep it concise — do not pad with fabricated content

Return ONLY the rewritten CV text, nothing else.

JOB DESCRIPTION:
{jd_text}

ORIGINAL CV:
{cv_text}
"""

def tailor_cv(cv_text: str, jd_text: str) -> str:
    import time
    from google.genai import errors

    client = _get_client()
    prompt = ATS_REWRITE_PROMPT.format(jd_text=jd_text, cv_text=cv_text)

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=config.REWRITE_MODEL,
                contents=prompt,
            )
            return response.text
        except errors.ServerError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s


def export_resume_to_pdf(resume_text: str, candidate_name: str = "Candidate") -> bytes:
    """
    Converts plain text ATS tailored resume into a clean, professional, print-ready PDF.
    """
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#C2410C")
    c_text = colors.HexColor("#1A1A1E")
    c_muted = colors.HexColor("#6B6660")
    c_border = colors.HexColor("#E7E0D5")

    title_style = ParagraphStyle(
        "CVTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=4,
    )

    h1_style = ParagraphStyle(
        "CVH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "CVBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_text,
        spaceAfter=2,
    )

    story = []
    story.append(Paragraph(candidate_name, title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=2, spaceAfter=8))

    for line in resume_text.split("\n"):
        clean = line.strip()
        if not clean:
            story.append(Spacer(1, 3))
            continue

        clean_no_md = clean.replace("**", "").replace("##", "").strip()
        is_heading = (
            clean.startswith("##")
            or (clean.isupper() and len(clean) < 35)
            or (clean_no_md.isupper() and len(clean_no_md) < 35 and not clean.startswith("-") and not clean.startswith("•"))
        )

        if is_heading:
            story.append(Paragraph(clean_no_md, h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=1, spaceAfter=3))
        elif clean.startswith("-") or clean.startswith("•") or clean.startswith("*"):
            bullet_text = clean.lstrip("-•* ").strip()
            story.append(
                Paragraph(
                    f"&bull; {bullet_text}",
                    ParagraphStyle("BulletItem", parent=body_style, leftIndent=12, spaceAfter=2),
                )
            )
        else:
            story.append(Paragraph(clean, body_style))

    doc.build(story)
    return buffer.getvalue()
