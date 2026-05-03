from __future__ import annotations

from pathlib import Path

from state.shared_state import load_state, save_state
from agents.member3_risk_scoring import run_member3
from agents.member4_report_agent import member4_ranking_report_node
from orchestration.integrate_member3_to_member4 import adapt_member3_output_for_member4


REQUIRED_SKILLS  = ["Python", "SQL", "Docker", "REST APIs"]
PREFERRED_SKILLS = ["Kubernetes", "AWS", "FastAPI"]
REQUIRED_YEARS   = 3


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Load whatever crew.py already saved (parsed candidates, fit analyses)
    existing_state = load_state()

    # ── Candidates come from parsed_candidates written by Member 1 in crew.py
    # Fall back to the candidates key if parsed_candidates is empty
    candidates = (
        existing_state.get("parsed_candidates")
        or existing_state.get("candidates")
        or []
    )

    if not candidates:
        raise RuntimeError(
            "No candidates found in state. "
            "Run crew.py first so Member 1 parses the resumes."
        )

    print(f"[app_pipeline] Loaded {len(candidates)} candidate(s) from state.")

    state = {
        "job_description": existing_state.get(
            "job_description",
            "Backend Developer role requiring Python, SQL, Docker, and 3+ years experience.",
        ),
        "job_requirements": existing_state.get(
            "job_requirements",
            {
                "required_skills":  REQUIRED_SKILLS,
                "preferred_skills": PREFERRED_SKILLS,
                "minimum_experience": REQUIRED_YEARS,
            },
        ),
        # ── Use candidates already parsed by Member 1
        "candidates":           candidates,
        "parsed_candidates":    existing_state.get("parsed_candidates", []),
        # ── Carry forward fit analyses from Member 2 if available
        "fit_analyses":         existing_state.get("fit_analyses", []),
        "scored_candidates":    [],
        "scoring_results":      [],
        "ranked_candidates":    [],
        "shortlisted_candidates": [],
        "ranking_summary":      {},
        "final_report":         "",
        "report_metadata":      {},
        "execution_trace":      existing_state.get("execution_trace", []),
    }

    save_state(state)

    # ── Member 3 — Risk & Scoring
    run_member3(
        required_skills  = REQUIRED_SKILLS,
        preferred_skills = PREFERRED_SKILLS,
        required_years   = REQUIRED_YEARS,
    )

    # ── Member 4 — Ranking & Report
    state = load_state()
    state = adapt_member3_output_for_member4(state)
    save_state(state)

    state = member4_ranking_report_node(state, model_name="qwen2.5:7b", top_n=2)
    save_state(state)

    print("\n=== Ranked Candidates ===")
    for candidate in state["ranked_candidates"]:
        print(
            f"Rank {candidate['rank']}: "
            f"{candidate['name']} — Score {candidate['score']} — "
            f"Risk {candidate.get('risk_level', 'Unknown')}"
        )

    print("\n=== Shortlisted Candidates ===")
    for candidate in state["shortlisted_candidates"]:
        print(f"Rank {candidate['rank']}: {candidate['name']}")

    print("\n=== Ranking Summary ===")
    print(state["ranking_summary"])

    print("\n=== Final Report ===")
    print(state["final_report"])

    print("\n=== Report Metadata ===")
    print(state["report_metadata"])

    print("\n=== Execution Trace Count ===")
    print(len(state.get("execution_trace", [])))


if __name__ == "__main__":
    main()