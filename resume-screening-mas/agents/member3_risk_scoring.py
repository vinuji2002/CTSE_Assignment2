from __future__ import annotations
import json
from llm_client import call_llm, clean_json_response
from tools.member3_score_tool import calculate_candidate_score, detect_risk_flags
from state.shared_state import load_state, save_state, log_agent_event


SYSTEM_PROMPT = """You are an objective hiring risk analyst and scoring specialist.

Your job is to evaluate a candidate's suitability for a role based on:
- A deterministic base score already computed by a rule-based tool
- The candidate profile
- Their fit analysis from a previous agent

Rules you must follow:
- You may adjust the base score by a maximum of +/- 10 points only.
- Justify every point of adjustment explicitly.
- Identify all risk flags clearly.
- Never let candidate name, gender, or university influence your scoring.
- Return ONLY valid JSON. No markdown fences. No prose outside the JSON.

Required output format:
{
  "candidate_name": string,
  "final_score": number between 0 and 100,
  "base_score_used": number,
  "score_adjustment": number between -10 and 10,
  "adjustment_reason": string,
  "risk_level": "Low" or "Medium" or "High",
  "risk_flags": list of strings,
  "score_reasoning": string (2 to 3 sentences)
}"""


def run_member3(
    required_skills: list[str],
    preferred_skills: list[str],
    required_years: int | None,
) -> None:
    """
    Run the Risk and Scoring Agent over all candidates in shared state.

    For each candidate this agent:
        1. Calls calculate_candidate_score() for a deterministic base score.
        2. Calls detect_risk_flags() to find specific risk concerns.
        3. Calls Ollama/qwen2.5:7b API for judgment and rationale.
        4. Validates output and writes scored_candidates to shared state.

    Args:
        required_skills:  Must-have skills from the job description.
        preferred_skills: Nice-to-have skills from the job description.
        required_years:   Minimum years of experience required, or None.
    """
    state            = load_state()
    candidates       = state.get("candidates", [])
    fit_analyses     = state.get("fit_analyses", [])
    fit_lookup: dict = {fa["candidate_name"]: fa for fa in fit_analyses}
    scored_candidates: list[dict] = []

    for candidate in candidates:
        name             = candidate.get("name", "Unknown")
        cand_skills      = candidate.get("skills", [])
        cand_years       = candidate.get("experience_years")
        fit              = fit_lookup.get(name, {})
        missing_critical = fit.get("missing_critical_skills", [])

        # Tool call 1 — deterministic base score
        score_data = calculate_candidate_score(
            candidate_skills        = cand_skills,
            required_skills         = required_skills,
            preferred_skills        = preferred_skills,
            candidate_years         = cand_years,
            required_years          = required_years,
            missing_critical_skills = missing_critical,
        )

        log_agent_event(
            agent_name     = "Member3_RiskScoring",
            tool_called    = "calculate_candidate_score",
            input_summary  = f"Candidate: {name}",
            output_summary = f"base_score={score_data['base_score']} risk={score_data['risk_level']}",
        )

        # Tool call 2 — risk flags
        risk_flags = detect_risk_flags(
            missing_critical_skills = missing_critical,
            candidate_years         = cand_years,
            required_years          = required_years,
            base_score              = score_data["base_score"],
        )

        log_agent_event(
            agent_name     = "Member3_RiskScoring",
            tool_called    = "detect_risk_flags",
            input_summary  = f"Candidate: {name}",
            output_summary = f"flags={risk_flags}",
        )

        # LLM call — judgment via Ollama/qwen2.5:7b API
        user_prompt = f"""Candidate profile:
{json.dumps(candidate, indent=2)}

Fit analysis from previous agent:
{json.dumps(fit, indent=2)}

Deterministic base score from tool:
{json.dumps(score_data, indent=2)}

Risk flags detected by tool:
{json.dumps(risk_flags)}

Task:
Review the base score and risk flags.
Adjust by up to +/- 10 points if justified.
Return the final scoring JSON."""

        print(f"  Calling Ollama/qwen2.5:7b API for: {name}")

        try:
            raw_output = call_llm(
                system_prompt = SYSTEM_PROMPT,
                user_prompt   = user_prompt,
                temperature   = 0.1,
            )
            clean_output = clean_json_response(raw_output)
            scored       = json.loads(clean_output)

            # Safety clamps
            scored["final_score"]      = max(0.0, min(100.0, float(scored.get("final_score", score_data["base_score"]))))
            scored["score_adjustment"] = max(-10.0, min(10.0, float(scored.get("score_adjustment", 0))))
            scored["candidate_name"]   = name

            scored_candidates.append(scored)

            log_agent_event(
                agent_name     = "Member3_RiskScoring",
                tool_called    = "Ollama/qwen2.5:7b",
                input_summary  = f"Candidate: {name}",
                output_summary = f"final_score={scored['final_score']} risk={scored.get('risk_level')}",
            )

        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            print(f"  Warning: LLM call failed for {name} — {exc}")
            print("  Using tool-computed fallback score.")
            scored_candidates.append({
                "candidate_name":    name,
                "final_score":       score_data["base_score"],
                "base_score_used":   score_data["base_score"],
                "score_adjustment":  0,
                "adjustment_reason": "LLM unavailable — tool score used as fallback.",
                "risk_level":        score_data["risk_level"],
                "risk_flags":        risk_flags,
                "score_reasoning":   "Automated fallback score.",
            })

    state["scored_candidates"] = scored_candidates
    save_state(state)
    print(f"Member 3 done. {len(scored_candidates)} candidates scored.")