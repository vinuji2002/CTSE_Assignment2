"""
agents/member1_resume_intelligence.py
Member 1 — Resume Intelligence Agent
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import requests

from state.shared_state import load_state, save_state, log_agent_event
from tools.member1_extract_resume import (
    extract_candidate_fields,
    normalise_skills,
    validate_candidate_profile,
)

logger = logging.getLogger("Member1_ResumeAgent")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"
AGENT_NAME = "Member1_ResumeAgent"
UPLOAD_DIR = Path("data/resumes")


# ─────────────────────────────────────────────────────────────
# PDF path resolution
# ─────────────────────────────────────────────────────────────

def _resolve_pdf_path(raw: dict[str, Any]) -> Path | None:
    """
    Return the Path to a candidate's PDF using multiple fallbacks.

    Fallback order
    ──────────────
    a) raw["path"]  — stored during upload (may be "" or stale)
    b) UPLOAD_DIR / raw["file"]  — reconstruct from filename key
    c) UPLOAD_DIR / <name_slug>_cv.pdf  — derive from candidate name
    d) Fuzzy — any PDF in UPLOAD_DIR whose stem contains the name words
    """
    # a) Explicit path stored in state
    stored = raw.get("path", "")
    if stored:
        p = Path(stored)
        if p.exists() and p.is_file():
            return p

    # b) Reconstruct from "file" key
    file_key = raw.get("file", "")
    if file_key:
        p = UPLOAD_DIR / file_key
        if p.exists():
            return p

    # c) Derive from candidate name
    name = raw.get("name", "")
    if name and name not in ("Unknown", ""):
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        for suffix in (f"{slug}_cv.pdf", f"{slug}.pdf", f"{slug}_resume.pdf"):
            p = UPLOAD_DIR / suffix
            if p.exists():
                return p

    # d) Fuzzy match — stem must contain every significant word in the name
    if name and UPLOAD_DIR.exists():
        words = [w.lower() for w in name.split() if len(w) > 2]
        for pdf in sorted(UPLOAD_DIR.glob("*.pdf")):
            stem = pdf.stem.lower()
            if words and all(w in stem for w in words):
                logger.info("  Fuzzy-matched '%s' → %s", name, pdf.name)
                return pdf

    return None


# ─────────────────────────────────────────────────────────────
# Sync disk PDFs into state — deduplicates by file AND name
# ─────────────────────────────────────────────────────────────

def _sync_candidates_from_disk(state: dict[str, Any]) -> dict[str, Any]:
    """
    Walk data/resumes/ and ensure every PDF appears in state["candidates"]
    with a correct, non-empty path.

    Deduplication strategy
    ──────────────────────
    * Match first by filename ("file" key) — fastest, most reliable.
    * If no file match, try matching by candidate name (case-insensitive).
      This handles entries created by app.py that had no "file" key.
      In that case the existing entry is patched with file + path rather
      than creating a second entry.
    * Only truly new PDFs (no match by file or name) get appended.

    This prevents the double-candidate bug where app.py registers a
    candidate without a "file" key and Member 1 then adds a second entry
    for the same PDF it finds on disk.
    """
    if not UPLOAD_DIR.exists():
        return state

    existing = state.setdefault("candidates", [])

    # Build lookup tables
    by_file: dict[str, dict] = {
        c.get("file", ""): c
        for c in existing
        if c.get("file")
    }
    by_name: dict[str, dict] = {
        c.get("name", "").lower(): c
        for c in existing
        if c.get("name")
    }

    for pdf in sorted(UPLOAD_DIR.glob("*.pdf")):
        if pdf.name in by_file:
            # Already tracked — just patch path to keep it current
            by_file[pdf.name]["path"] = str(pdf.resolve())
            continue

        # Derive display name from filename
        pretty = pdf.stem.replace("_", " ").replace("-", " ").title()
        pretty = re.sub(r"\s+(Cv|Resume)$", "", pretty, flags=re.IGNORECASE)

        existing_entry = by_name.get(pretty.lower())
        if existing_entry is not None:
            # Candidate already in state (added by app.py without "file" key)
            # — patch the missing fields instead of adding a duplicate
            existing_entry.setdefault("file", pdf.name)
            existing_entry["path"] = str(pdf.resolve())
            by_file[pdf.name] = existing_entry
            logger.info(
                "  Patched path for existing candidate '%s' → %s",
                pretty, pdf.name,
            )
        else:
            # Genuinely new PDF not yet in state
            entry = {"name": pretty, "file": pdf.name, "path": str(pdf.resolve())}
            existing.append(entry)
            by_file[pdf.name] = entry
            by_name[pretty.lower()] = entry
            logger.info("  Discovered new PDF: %s", pdf.name)

    state["candidates"] = existing
    return state


# ─────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────

def _read_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF. Tries pdfplumber, falls back to pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            logger.debug("  pdfplumber: %d chars from %s", len(text), pdf_path.name)
            return text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("  pdfplumber failed for %s: %s", pdf_path.name, exc)

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        logger.debug("  pypdf: %d chars from %s", len(text), pdf_path.name)
        return text
    except Exception as exc:
        logger.error("  pypdf also failed for %s: %s", pdf_path.name, exc)
        return ""


# ─────────────────────────────────────────────────────────────
# Ollama LLM helpers
# ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    payload = {
        "model":    MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama at localhost:11434. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc


def _llm_summary(name: str, skills: list[str], years: int | None) -> str:
    skills_str = ", ".join(skills) if skills else "none listed"
    years_str  = str(years) if years is not None else "unknown"
    prompt = (
        f"You are an HR assistant reviewing a resume.\n\n"
        f"Candidate : {name}\n"
        f"Skills    : {skills_str}\n"
        f"Experience: {years_str} years\n\n"
        "Write ONE concise sentence (max 30 words) summarising this candidate's "
        "professional profile for a recruiter. Do not add any preamble or label."
    )
    return _call_ollama(prompt).strip()


# ─────────────────────────────────────────────────────────────
# Public agent entry-point
# ─────────────────────────────────────────────────────────────

def run_member1(job_requirements: dict[str, Any] | None = None) -> None:
    """
    Load shared state → sync disk PDFs → extract fields from each PDF
    → enrich with LLM → write parsed_candidates back to shared state.
    """
    state = load_state()

    # Patch correct paths; deduplicates against existing state entries
    state = _sync_candidates_from_disk(state)
    save_state(state)

    raw_candidates: list[dict[str, Any]] = state.get("candidates", [])
    if not raw_candidates:
        logger.warning("No candidates in shared state — Member 1 skipped.")
        state["parsed_candidates"] = []
        save_state(state)
        return

    job_req = job_requirements or state.get("job_requirements", {})
    parsed:  list[dict[str, Any]] = []

    for raw in raw_candidates:
        name = raw.get("name", "Unknown")
        logger.info("Processing candidate: %s", name)

        # Resolve PDF path (handles empty / stale path)
        pdf_path = _resolve_pdf_path(raw)

        if pdf_path is None:
            logger.warning(
                "  Cannot locate PDF for '%s' "
                "(path=%r, file=%r) — will use existing skills if available.",
                name, raw.get("path", ""), raw.get("file", ""),
            )
            raw_text = ""
        else:
            logger.info("  Reading: %s", pdf_path)
            raw_text = _read_pdf_text(pdf_path)
            if not raw_text:
                logger.warning("  Empty text from %s.", pdf_path.name)

        raw_with_text = {**raw, "raw_text": raw_text}

        # Step 1: deterministic extraction
        fields           = extract_candidate_fields(raw_with_text)
        fields["skills"] = normalise_skills(fields["skills"])

        # Fallback to existing skills list if PDF gave nothing
        if not fields["skills"] and raw.get("skills"):
            fields["skills"] = normalise_skills(
                [str(s) for s in raw["skills"]]
            )
            logger.info(
                "  Using pre-existing skills for '%s' (PDF yielded none).", name
            )

        # Preserve experience_years from raw if extraction missed it
        if fields["experience_years"] is None and raw.get("experience_years") is not None:
            fields["experience_years"] = raw["experience_years"]

        # Step 2: validate
        validation = validate_candidate_profile(fields, job_req)

        # Step 3: LLM enrichment
        try:
            summary = _llm_summary(name, fields["skills"], fields["experience_years"])
        except RuntimeError as exc:
            logger.error("  LLM call failed for %s: %s", name, exc)
            summary = "Summary unavailable — LLM offline."

        # Step 4: assemble parsed candidate
        candidate_skills_lower = [s.lower() for s in fields["skills"]]
        missing_critical       = [
            s for s in job_req.get("required_skills", [])
            if s.lower() not in candidate_skills_lower
        ]

        parsed_candidate: dict[str, Any] = {
            "candidate_id":            f"cand_{len(parsed)+1:03d}",
            "name":                    fields["name"],
            "file":                    fields.get("file") or raw.get("file", ""),
            "email":                   fields.get("email", ""),
            "skills":                  fields["skills"],
            "experience_years":        fields["experience_years"],
            "education":               fields["education"],
            "validation_passed":       validation["valid"],
            "validation_notes":        validation["notes"],
            "match_score":             validation["match_score"],
            "profile_summary":         summary,
            "missing_critical_skills": missing_critical,
        }

        parsed.append(parsed_candidate)

        log_agent_event(
            agent_name     = AGENT_NAME,
            tool_called    = "extract_candidate_fields",
            input_summary  = (
                f"candidate={name}, "
                f"pdf={pdf_path.name if pdf_path else 'NOT_FOUND'}, "
                f"chars={len(raw_text)}"
            ),
            output_summary = (
                f"skills={fields['skills']}, "
                f"years={parsed_candidate['experience_years']}, "
                f"match={validation['match_score']}"
            ),
        )

    state["parsed_candidates"] = parsed
    save_state(state)
    logger.info("Member 1 complete — %d candidates parsed.", len(parsed))


# ─────────────────────────────────────────────────────────────
# Standalone smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")

    pdfs = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs in data/resumes/ — run generate_sample_cvs.py first.")
    else:
        for pdf in pdfs:
            text   = _read_pdf_text(pdf)
            raw    = {
                "name":     pdf.stem.replace("_", " ").title(),
                "file":     pdf.name,
                "path":     str(pdf.resolve()),
                "raw_text": text,
            }
            fields = extract_candidate_fields(raw)
            fields["skills"] = normalise_skills(fields["skills"])
            print(f"\n{'─'*52}")
            print(f"File   : {pdf.name}")
            print(f"Name   : {fields['name']}")
            print(f"Skills : {fields['skills']}")
            print(f"Years  : {fields['experience_years']}")
            print(f"Email  : {fields['email']}")
            print(f"Edu    : {fields['education']}")