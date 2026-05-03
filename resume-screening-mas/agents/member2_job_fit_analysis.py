"""
agents/member2_fit_agent.py
Member 2 — Job Fit Analysis Agent

Responsibilities:
  - Read parsed_candidates and job_requirements from shared state
  - Use deterministic tools to compute structural skill/experience matches
  - Call the LLM (qwen2.5:7b via Ollama HTTP API) to produce fit reasoning
  - Write fit_analyses back to shared state

Model: qwen2.5:7b  (local Ollama)
Tool:  tools/member2_parse_jd.py  (already provided — parse_job_description,
                                    match_candidate_to_job)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from state.shared_state import load_state, save_state, log_agent_event
from tools.member2_parse_jd import parse_job_description, match_candidate_to_job

logger = logging.getLogger("Member2_FitAgent")

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL_NAME  = "qwen2.5:7b"
AGENT_NAME  = "Member2_FitAgent"


# ─────────────────────────────────────────────────────────────
# Internal LLM helper
# ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama server and return the text reply.

    Raises:
        RuntimeError: If Ollama is not reachable or the call fails.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama at localhost:11434. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────
# LLM fit reasoning for a single candidate
# ─────────────────────────────────────────────────────────────

def _generate_fit_reasoning(
    candidate_name: str,
    matched_skills: list[str],
    missing_critical_skills: list[str],
    experience_gap: bool,
    job_description: str,
) -> str:
    """
    Ask the LLM to write a concise fit-reasoning sentence for this candidate.

    Returns a plain-text reasoning string (1–2 sentences).
    """
    matched_str = ", ".join(matched_skills) if matched_skills else "none"
    missing_str = ", ".join(missing_critical_skills) if missing_critical_skills else "none"
    exp_note    = "does NOT meet" if experience_gap else "meets"

    prompt = (
        f"You are an HR analyst evaluating job candidates.\n\n"
        f"Job description summary:\n{job_description[:300]}\n\n"
        f"Candidate        : {candidate_name}\n"
        f"Matched skills   : {matched_str}\n"
        f"Missing skills   : {missing_str}\n"
        f"Experience       : candidate {exp_note} the minimum requirement\n\n"
        f"Write ONE or TWO concise sentences explaining how well this candidate "
        f"fits the role. Be objective. Do not add any preamble or labels."
    )

    return _call_ollama(prompt).strip()


# ─────────────────────────────────────────────────────────────
# Public agent entry-point
# ─────────────────────────────────────────────────────────────

def run_member2() -> None:
    """
    Load shared state, analyse every parsed candidate against the job,
    write fit_analyses back to shared state.

    Called by the pipeline runner (app_pipeline.py).
    """
    state = load_state()

    job_description: str              = state.get("job_description", "")
    job_requirements: dict[str, Any]  = state.get("job_requirements", {})

    # Prefer parsed_candidates if Member 1 has already run; fall back to
    # raw candidates so Member 2 can still run standalone.
    candidates: list[dict[str, Any]] = (
        state.get("parsed_candidates")
        or state.get("candidates", [])
    )

    if not candidates:
        logger.warning("No candidates found in shared state — skipping Member 2.")
        state["fit_analyses"] = []
        save_state(state)
        return

    # If job_requirements is sparse, try to enrich from job_description text
    if not job_requirements.get("required_skills") and job_description:
        logger.info("job_requirements empty — parsing from job_description text.")
        job_requirements = parse_job_description(job_description)
        state["job_requirements"] = job_requirements

    fit_analyses: list[dict[str, Any]] = []

    for candidate in candidates:
        name = candidate.get("name", "Unknown")
        logger.info("Analysing fit for candidate: %s", name)

        # ── Step 1: deterministic structural match ─────────────
        match_result = match_candidate_to_job(candidate, job_requirements)

        matched_skills           = match_result.get("matched_skills", [])
        missing_critical_skills  = match_result.get("missing_critical_skills", [])
        experience_gap           = match_result.get("experience_gap", False)

        # ── Step 2: LLM fit reasoning ──────────────────────────
        try:
            fit_reasoning = _generate_fit_reasoning(
                candidate_name          = name,
                matched_skills          = matched_skills,
                missing_critical_skills = missing_critical_skills,
                experience_gap          = experience_gap,
                job_description         = job_description,
            )
        except RuntimeError as exc:
            logger.error("LLM call failed for %s: %s", name, exc)
            fit_reasoning = "Fit reasoning unavailable — LLM offline."

        # ── Step 3: classify overall fit level ────────────────
        if not missing_critical_skills and not experience_gap:
            fit_level = "Strong"
        elif len(missing_critical_skills) <= 1 and not experience_gap:
            fit_level = "Moderate"
        else:
            fit_level = "Weak"

        # ── Step 4: assemble fit analysis entry ───────────────
        fit_entry: dict[str, Any] = {
            "candidate_name":          name,
            "matched_skills":          matched_skills,
            "missing_critical_skills": missing_critical_skills,
            "experience_gap":          experience_gap,
            "fit_level":               fit_level,
            "fit_reasoning":           fit_reasoning,
        }

        fit_analyses.append(fit_entry)

        log_agent_event(
            agent_name     = AGENT_NAME,
            tool_called    = "match_candidate_to_job",
            input_summary  = f"candidate={name}",
            output_summary = (
                f"fit_level={fit_level}, "
                f"matched={len(matched_skills)}, "
                f"missing={len(missing_critical_skills)}"
            ),
        )

    state["fit_analyses"] = fit_analyses
    save_state(state)

    logger.info(
        "Member 2 complete — %d fit analyses saved.", len(fit_analyses)
    )