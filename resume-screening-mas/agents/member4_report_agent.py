from __future__ import annotations

from typing import Any

from ollama import chat

from state.shared_state import MASState
from tools.report_logger_tool import (
    append_trace_to_state,
    build_ranking_summary,
    generate_shortlist,
    log_agent_trace_to_file,
    rank_candidates,
    save_ranked_candidates,
    save_ranking_summary,
    save_report,
    save_shortlist,
)

AGENT_NAME = "Member4_Ranking_Report_Agent"

SYSTEM_PROMPT = """
You are the Ranking and Report Agent in a locally hosted multi-agent hiring system.

Your responsibilities:
1. Review candidate scoring results provided by previous agents.
2. Rank candidates from strongest to weakest.
3. Recommend a shortlist.
4. Write a concise recruiter-friendly final report.
5. Do not invent facts.
6. Use only the candidate data provided.
7. Clearly mention strengths, weaknesses, and final recommendation.
8. Keep the report factual, structured, and concise.
9. Do not repeat the same point unnecessarily.

Return plain text only.
""".strip()


def build_candidate_snapshot(candidate: dict[str, Any]) -> str:
    """
    Convert candidate information into a clean text block for the prompt.

    Args:
        candidate: Candidate dictionary.

    Returns:
        A structured text description of the candidate.
    """
    return (
        f"Rank: {candidate.get('rank', 'N/A')}\n"
        f"Name: {candidate.get('name', 'Unknown')}\n"
        f"Score: {candidate.get('score', 'Unknown')}\n"
        f"Risk level: {candidate.get('risk_level', 'Unknown')}\n"
        f"Risk flags: {candidate.get('risk_flags', [])}\n"
        f"Fit reasoning: {candidate.get('fit_reasoning', 'Not available')}\n"
        f"Score reasoning: {candidate.get('score_reasoning', 'Not available')}\n"
    )


def build_report_prompt(
    job_description: str,
    ranked_candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    ranking_summary: dict[str, Any],
    top_candidate: str,
    shortlisted_names: list[str],
    less_competitive_candidate: str,
) -> str:
    """
    Build the user prompt sent to the LLM for final report generation.

    Args:
        job_description: Original job description.
        ranked_candidates: Full ranked candidate list.
        shortlist: Top shortlisted candidates.
        ranking_summary: Ranking metadata summary.
        top_candidate: Name of the highest-ranked candidate.
        shortlisted_names: Names of shortlisted candidates.
        less_competitive_candidate: Name of the lowest-ranked candidate.

    Returns:
        Prompt text for the LLM.
    """
    ranked_blocks = "\n\n".join(
        build_candidate_snapshot(candidate) for candidate in ranked_candidates
    )
    shortlist_names_text = ", ".join(shortlisted_names)

    return f"""
Job description:
{job_description}

Ranking summary:
{ranking_summary}

Top candidate:
{top_candidate}

Official shortlisted candidates:
{shortlist_names_text}

Lowest-ranked candidate currently less competitive:
{less_competitive_candidate}

Only these candidates are shortlisted. Do not include any other candidate in the shortlisted candidates section.

Candidate details:
{ranked_blocks}

Task:
Write a final hiring report for the recruiter.

The final recommendation must follow these exact decisions:
- Strongest candidate: {top_candidate}
- Candidates proceeding to the next stage: {shortlist_names_text}
- Lower-ranked candidate currently less competitive: {less_competitive_candidate}

Use exactly these headings:
1. Role summary
2. Candidate ranking
3. Shortlisted candidates
4. Key strengths
5. Key risks or weaknesses
6. Final recommendation

Section rules:
- In "Candidate ranking", discuss all ranked candidates.
- In "Shortlisted candidates", mention only the candidates listed in the official shortlisted candidates input.
- Do not describe any non-shortlisted candidate as shortlisted.
- In "Key strengths", focus mainly on shortlisted candidates.
- In "Key risks or weaknesses", mention notable concerns supported by the input.
- In "Final recommendation", the highest-ranked candidate MUST be identified as the strongest candidate.
- Only shortlisted candidates may be recommended to proceed.
- Candidates not shortlisted MUST be described as less competitive.
- Do not contradict the ranking, shortlist, or scores under any circumstance.

Rules:
- Be concise and professional.
- Use only the facts provided.
- Mention candidate rank numbers.
- Explain why the shortlisted candidates were selected.
- Mention notable risks briefly.
- Do not add information that is not in the input.
- Do not use markdown tables.
- Use plain text bullet points or numbered lines for rankings.
- Keep the final recommendation strict and evidence-based.
- Do not speculate about personality, adaptability, eagerness to learn, future growth, or training potential unless explicitly provided in the input.
""".strip()


def generate_fallback_report(
    job_description: str,
    ranked_candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
) -> str:
    """
    Generate a safe fallback report if the Ollama call fails.

    Args:
        job_description: Original job description.
        ranked_candidates: Full ranked candidate list.
        shortlist: Top shortlisted candidates.

    Returns:
        Plain text fallback report.
    """
    ranking_lines: list[str] = []
    for candidate in ranked_candidates:
        ranking_lines.append(
            f"- Rank {candidate.get('rank', 'N/A')}: "
            f"{candidate.get('name', 'Unknown')} "
            f"(Score: {candidate.get('score', 'Unknown')}, "
            f"Risk: {candidate.get('risk_level', 'Unknown')})"
        )

    shortlist_lines: list[str] = []
    for candidate in shortlist:
        shortlist_lines.append(
            f"- Rank {candidate.get('rank', 'N/A')}: "
            f"{candidate.get('name', 'Unknown')} "
            f"(Score: {candidate.get('score', 'Unknown')}, "
            f"Risk: {candidate.get('risk_level', 'Unknown')})"
        )

    strongest_candidate = ranked_candidates[0]["name"] if ranked_candidates else "No candidate"
    next_candidate = shortlist[1]["name"] if len(shortlist) > 1 else "No additional candidate"
    less_competitive_candidate = (
        ranked_candidates[-1]["name"] if ranked_candidates else "No candidate"
    )

    return "\n".join(
        [
            "1. Role summary",
            job_description or "No job description provided.",
            "",
            "2. Candidate ranking",
            *ranking_lines,
            "",
            "3. Shortlisted candidates",
            *shortlist_lines,
            "",
            "4. Key strengths",
            "The shortlisted candidates show the strongest overall alignment based on score and fit reasoning.",
            "",
            "5. Key risks or weaknesses",
            "Review the documented risk flags and missing skills before final interview selection.",
            "",
            "6. Final recommendation",
            (
                f"{strongest_candidate} should proceed as the strongest candidate. "
                f"{next_candidate} should also proceed to the next stage if applicable. "
                f"{less_competitive_candidate} is currently less competitive based on the available scoring results."
            ),
        ]
    )


def clean_report_text(report: str) -> str:
    """
    Light cleanup for model output to reduce unsupported speculative phrasing.

    Args:
        report: Raw report text from the model.

    Returns:
        Cleaned report text.
    """
    replacements = {
        "adaptability and eagerness to learn": "available scoring results",
        "future growth": "current scoring results",
        "training potential": "identified technical gap",
    }

    cleaned = report
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    return cleaned.strip()


def generate_report_text(
    job_description: str,
    ranked_candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    ranking_summary: dict[str, Any],
    model_name: str = "qwen2.5:7b",
) -> str:
    """
    Call the Ollama model to generate the final recruiter report.

    Args:
        job_description: Original job description.
        ranked_candidates: Full ranked candidate list.
        shortlist: Top shortlisted candidates.
        ranking_summary: Ranking metadata summary.
        model_name: Local Ollama model name.

    Returns:
        Final report text.
    """
    top_candidate = ranked_candidates[0]["name"] if ranked_candidates else "Unknown"
    shortlisted_names = [candidate.get("name", "Unknown") for candidate in shortlist]
    less_competitive_candidate = ranked_candidates[-1]["name"] if ranked_candidates else "Unknown"

    prompt = build_report_prompt(
        job_description=job_description,
        ranked_candidates=ranked_candidates,
        shortlist=shortlist,
        ranking_summary=ranking_summary,
        top_candidate=top_candidate,
        shortlisted_names=shortlisted_names,
        less_competitive_candidate=less_competitive_candidate,
    )

    try:
        response = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return clean_report_text(response["message"]["content"])
    except Exception:
        return generate_fallback_report(job_description, ranked_candidates, shortlist)


def member4_ranking_report_node(
    state: MASState,
    model_name: str = "qwen2.5:7b",
    top_n: int = 3,
) -> MASState:
    """
    Member 4 agent node.
    Reads scoring_results from shared state, ranks candidates, generates
    a shortlist, calls the LLM for a final report, updates shared state,
    saves outputs, and logs execution trace.

    Args:
        state: Shared global MAS state.
        model_name: Ollama model used for report generation.
        top_n: Number of candidates to include in the shortlist.

    Returns:
        Updated shared state.

    Raises:
        KeyError: If scoring_results is missing from the state.
        ValueError: If scoring_results is empty or top_n is invalid.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if "scoring_results" not in state:
        raise KeyError("Shared state must contain 'scoring_results'")

    scoring_results = state["scoring_results"]

    if not scoring_results:
        raise ValueError("No scoring results found for Member 4 to rank")

    ranked_candidates = rank_candidates(scoring_results)
    shortlist = generate_shortlist(ranked_candidates, top_n=top_n)
    ranking_summary = build_ranking_summary(ranked_candidates, shortlist)

    final_report = generate_report_text(
        job_description=state.get("job_description", ""),
        ranked_candidates=ranked_candidates,
        shortlist=shortlist,
        ranking_summary=ranking_summary,
        model_name=model_name,
    )

    state["ranked_candidates"] = ranked_candidates
    state["shortlisted_candidates"] = shortlist
    state["ranking_summary"] = ranking_summary
    state["final_report"] = final_report
    state["report_metadata"] = {
        "model_name": model_name,
        "report_path": "outputs/final_report.txt",
        "ranked_candidates_path": "outputs/ranked_candidates.json",
        "shortlist_path": "outputs/shortlist.json",
        "ranking_summary_path": "outputs/ranking_summary.json",
        "trace_log_path": "outputs/trace_log.jsonl",
    }

    output_data = {
        "ranked_candidates": ranked_candidates,
        "shortlisted_candidates": shortlist,
        "ranking_summary": ranking_summary,
        "final_report": final_report,
        "report_metadata": state["report_metadata"],
    }

    save_report(final_report, output_path="outputs/final_report.txt")
    save_ranked_candidates(
        ranked_candidates, output_path="outputs/ranked_candidates.json"
    )
    save_shortlist(shortlist, output_path="outputs/shortlist.json")
    save_ranking_summary(
        ranking_summary, output_path="outputs/ranking_summary.json"
    )

    append_trace_to_state(
        state=state,
        agent_name=AGENT_NAME,
        input_data={
            "job_description": state.get("job_description", ""),
            "scoring_results": scoring_results,
            "model_name": model_name,
            "top_n": top_n,
        },
        output_data=output_data,
    )

    log_agent_trace_to_file(
        agent_name=AGENT_NAME,
        input_data={
            "job_description": state.get("job_description", ""),
            "scoring_results": scoring_results,
            "model_name": model_name,
            "top_n": top_n,
        },
        output_data=output_data,
        log_path="outputs/trace_log.jsonl",
    )

    return state