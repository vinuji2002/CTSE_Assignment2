from __future__ import annotations

import sys
from pathlib import Path

from state.shared_state import load_state, save_state
from agents.member1_resume_intelligence import run_member1
from agents.member2_job_fit_analysis import run_member2
from agents.member3_risk_scoring import run_member3
from agents.member4_report_agent import member4_ranking_report_node
from orchestration.integrate_member3_to_member4 import adapt_member3_output_for_member4


JD_PATH     = "data/jd.txt"
OUTPUT_PATH = "outputs/shortlist.md"


def _is_valid_candidate(c: dict) -> bool:
    """
    Filter out junk entries that crept in from the upload pipeline.

    A candidate is considered valid only if:
      - It has a real name (not a filename, not empty)
      - It has at least one skill  OR  a non-zero experience_years
    """
    name = (c.get("name") or "").strip()

    # Reject blank names
    if not name:
        return False

    # Reject entries that look like filenames  (contain _ followed by __ or end
    # with common extensions)
    filename_hints = ("__", ".pdf", ".docx", ".txt", ".csv", "cv_", "resume_")
    if any(h.lower() in name.lower() for h in filename_hints):
        return False

    # Reject entries whose name looks like a project / system title but has no
    # skills and zero experience  (e.g. "AI powered VR System")
    skills   = c.get("skills") or []
    exp      = c.get("experience_years") or 0
    if not skills and not exp:
        return False

    return True


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Load whatever state the upload app already wrote ────
    state = load_state()

    # ── Validate: candidates must exist before we proceed ───
    raw_candidates = state.get("candidates", [])
    if not raw_candidates:
        print(
            "\n[ERROR] No candidates found in shared state.\n"
            "  → Upload CVs via the web dashboard first (python app.py),\n"
            "    then re-run crew.py.\n"
        )
        sys.exit(1)

    # ── Strip junk / filename-only entries ──────────────────
    valid_candidates = [c for c in raw_candidates if _is_valid_candidate(c)]
    junk_count = len(raw_candidates) - len(valid_candidates)
    if junk_count:
        print(
            f"[INFO] Filtered out {junk_count} non-candidate entry/entries "
            f"(filenames or empty records) from shared state."
        )
    if not valid_candidates:
        print(
            "\n[ERROR] No valid candidates remain after filtering.\n"
            "  → Ensure uploaded CVs are parsed correctly by Member 1.\n"
        )
        sys.exit(1)

    state["candidates"] = valid_candidates

    # ── Load job description from disk if present ────────────
    jd_path = Path(JD_PATH)
    if jd_path.exists():
        state["job_description"] = jd_path.read_text(encoding="utf-8")
    elif not state.get("job_description"):
        print(
            f"[WARNING] {JD_PATH} not found and no job_description in state.\n"
            "  → Add a job description via the web dashboard or create data/jd.txt.\n"
        )

    # ── Validate: job requirements must exist ────────────────
    job_req = state.get("job_requirements", {})
    if not job_req.get("required_skills"):
        print(
            "\n[ERROR] No job requirements found in shared state.\n"
            "  → Configure required skills via the web dashboard first.\n"
        )
        sys.exit(1)

    required_skills  = job_req["required_skills"]
    preferred_skills = job_req.get("preferred_skills", [])
    required_years   = job_req.get("minimum_experience")

    # ── Reset downstream keys so re-runs are clean ──────────
    state.update({
        "parsed_candidates":      [],
        "fit_analyses":           [],
        "fit_results":            [],
        "scored_candidates":      [],
        "scoring_results":        [],
        "ranked_candidates":      [],
        "shortlisted_candidates": [],
        "ranking_summary":        {},
        "final_report":           "",
        "report_metadata":        {},
        "execution_trace":        [],
    })
    save_state(state)

    print(f"\n{'='*50}")
    print(f"MAS Pipeline — {len(valid_candidates)} candidate(s) loaded from state")
    print(f"  Required skills  : {required_skills}")
    print(f"  Preferred skills : {preferred_skills}")
    print(f"  Min experience   : {required_years} year(s)")
    print(f"{'='*50}")

    # ── Member 1 — Resume Intelligence Agent ────────────────
    print(f"\n{'='*50}")
    print("MEMBER 1 — Resume Intelligence Agent")
    print(f"{'='*50}")
    run_member1()
    state = load_state()
    for c in state.get("parsed_candidates", []):
        valid = c.get("validation_passed", "?")
        print(f"  {c['name']}: valid={valid}  summary={c.get('profile_summary', '—')[:80]}")

    # ── Member 2 — Job Fit Analysis Agent ───────────────────
    print(f"\n{'='*50}")
    print("MEMBER 2 — Job Fit Analysis Agent")
    print(f"{'='*50}")
    run_member2()
    state = load_state()

    # Sync fit_analyses → fit_results for the dashboard
    fit_analyses = state.get("fit_analyses", [])
    if fit_analyses and not state.get("fit_results"):
        state["fit_results"] = fit_analyses
        save_state(state)

    for r in state.get("fit_analyses", []):
        print(f"  {r['candidate_name']}: fit_level={r['fit_level']}")
        if r.get("missing_critical_skills"):
            print(f"    Missing   : {', '.join(r['missing_critical_skills'])}")
        if r.get("fit_reasoning"):
            print(f"    Reasoning : {r['fit_reasoning']}")

    # ── Member 3 — Risk & Scoring Agent ─────────────────────
    print(f"\n{'='*50}")
    print("MEMBER 3 — Risk and Scoring Agent")
    print(f"{'='*50}")
    run_member3(
        required_skills  = required_skills,
        preferred_skills = preferred_skills,
        required_years   = required_years,
    )
    state = load_state()
    for r in state.get("scored_candidates", []):
        print(f"  {r['candidate_name']}: score={r['final_score']}  risk={r['risk_level']}")
        for flag in r.get("risk_flags", []):
            print(f"    ⚠  {flag}")

    # ── Bridge: adapt Member 3 output for Member 4 ──────────
    state = adapt_member3_output_for_member4(state)
    save_state(state)

    # ── Member 4 — Ranking & Report Agent ───────────────────
    print(f"\n{'='*50}")
    print("MEMBER 4 — Ranking and Report Agent")
    print(f"{'='*50}")

    scored = state.get("scored_candidates", [])

    # Shortlist only Low / Medium risk candidates.
    # If everyone is High risk, fall back to the single top scorer
    # so the report is never completely empty.
    shortlist_eligible = [
        c for c in scored if c.get("risk_level") in ("Low", "Medium")
    ]
    if shortlist_eligible:
        top_n = len(shortlist_eligible)
        print(f"  Shortlisting {top_n} Low/Medium-risk candidate(s).")
    else:
        print(
            "  [WARN] All candidates scored High risk. "
            "Shortlisting top scorer as fallback."
        )
        top_n = 1

    state = member4_ranking_report_node(
        state,
        model_name="qwen2.5:7b",
        top_n=top_n,
    )
    save_state(state)

    # ── Print results (mirrors test_member4.py output) ───────
    shortlisted_names = {s["name"] for s in state.get("shortlisted_candidates", [])}

    print("\n=== Ranked Candidates ===")
    for c in state.get("ranked_candidates", []):
        tag = "✓ Shortlisted" if c["name"] in shortlisted_names else "✗ Not shortlisted"
        print(
            f"  Rank {c['rank']}: {c['name']}"
            f" — Score {c['score']}"
            f" — Risk {c.get('risk_level', 'Unknown')}"
            f" — {tag}"
        )

    print("\n=== Shortlisted Candidates ===")
    for c in state.get("shortlisted_candidates", []):
        print(f"  Rank {c['rank']}: {c['name']}")

    print("\n=== Ranking Summary ===")
    print(state.get("ranking_summary", {}))

    print("\n=== Final Report ===")
    print(state.get("final_report", ""))

    print("\n=== Report Metadata ===")
    print(state.get("report_metadata", {}))

    print("\n=== Execution Trace Count ===")
    print(len(state.get("execution_trace", [])))

    print(f"\n{'='*50}")
    print("DONE")
    print(f"  Report → {OUTPUT_PATH}")
    print(f"  Log    → logs/agent_trace.log")
    print(f"  State  → state/shared_state.json")
    print(f"{'='*50}")

def run_pipeline():
    main()

if __name__ == "__main__":
    main()