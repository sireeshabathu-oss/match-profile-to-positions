"""
Run this file to see the crew in action:

    python main.py

Edit CANDIDATE_PROFILE and JOB_POSITIONS below to try your own data.
"""

import json
import re

from crew import build_crew, calculate_match_score


CANDIDATE_PROFILE = """
Name: Priya Nair
Experience: 5 years
Skills: Python, SQL, Machine Learning, Data Visualization, A/B Testing
Summary: Data scientist with a strong background in retail demand forecasting
and building production machine learning pipelines.
"""

JOB_POSITIONS = """
1. Senior Data Scientist
   Required skills: Python, Machine Learning, SQL
   Nice-to-have skills: Data Visualization, A/B Testing
   Minimum experience: 4 years

2. Frontend Engineer
   Required skills: JavaScript, React
   Nice-to-have skills: CSS, TypeScript
   Minimum experience: 1 year
"""


def extract_json(text: str) -> dict:
    """Extract JSON from an LLM response."""

    text = text.strip()

    # Remove markdown code fences if the model used them.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def normalize_skills(skills: list[str]) -> set[str]:
    """Normalize skill names for comparison."""

    return {
        skill.strip().lower()
        for skill in skills
        if isinstance(skill, str) and skill.strip()
    }


def build_final_matches(candidate_data: dict, position_data: dict) -> list[dict]:
    """Calculate deterministic match scores from structured agent output."""

    candidate_skills = normalize_skills(candidate_data.get("skills", []))

    candidate_years = float(
        candidate_data.get("years_of_experience", 0) or 0
    )

    matches = []

    for position in position_data.get("positions", []):
        required_skills = position.get("required_skills", [])
        preferred_skills = position.get("preferred_skills", [])

        required_normalized = normalize_skills(required_skills)
        preferred_normalized = normalize_skills(preferred_skills)

        matching_required = candidate_skills & required_normalized
        matching_preferred = candidate_skills & preferred_normalized

        missing_required = required_normalized - candidate_skills

        required_years = float(
            position.get("minimum_years_experience", 0) or 0
        )

        score = calculate_match_score(
            required_skills=list(required_normalized),
            matching_required_skills=list(matching_required),
            preferred_skills=list(preferred_normalized),
            matching_preferred_skills=list(matching_preferred),
            candidate_years=candidate_years,
            required_years=required_years,
        )

        matches.append(
            {
                "position": position.get("title", "Unknown"),
                "match_score": score,
                "matching_required_skills": sorted(matching_required),
                "missing_required_skills": sorted(missing_required),
                "matching_preferred_skills": sorted(matching_preferred),
                "candidate_years": candidate_years,
                "required_years": required_years,
            }
        )

    return sorted(
        matches,
        key=lambda match: match["match_score"],
        reverse=True,
    )


if __name__ == "__main__":
    crew = build_crew(CANDIDATE_PROFILE, JOB_POSITIONS)

    result = crew.kickoff()

    # The first task analyzes the candidate.
    candidate_raw = result.tasks_output[0].raw

    # The second task analyzes the job positions.
    positions_raw = result.tasks_output[1].raw

    candidate_data = extract_json(candidate_raw)
    position_data = extract_json(positions_raw)

    final_matches = build_final_matches(
        candidate_data,
        position_data,
    )

    print("\n\n===== FINAL MATCH RESULTS =====\n")

    for match in final_matches:
        print(f"Position: {match['position']}")
        print(f"Match Score: {match['match_score']}/100")

        print(
            "Matching Required Skills:",
            ", ".join(match["matching_required_skills"])
            or "None",
        )

        print(
            "Missing Required Skills:",
            ", ".join(match["missing_required_skills"])
            or "None",
        )

        print(
            "Matching Preferred Skills:",
            ", ".join(match["matching_preferred_skills"])
            or "None",
        )

        print(
            f"Experience: {match['candidate_years']} years "
            f"(required: {match['required_years']} years)"
        )

        print("-" * 50)
