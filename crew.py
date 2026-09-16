"""
Defines the AI agents and tasks that make up the Match Profile to Positions crew.

Three agents work together in sequence:
1. Profile Analyst  - reads the candidate profile and extracts skills/experience
2. Position Analyst - reads the job postings and extracts requirements
3. Matchmaker       - compares the two and explains the match
"""

import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Which LLM to use.
MODEL = os.getenv("MODEL", "gpt-4o-mini")


def calculate_match_score(
    required_skills: list[str],
    matching_required_skills: list[str],
    preferred_skills: list[str],
    matching_preferred_skills: list[str],
    candidate_years: float,
    required_years: float,
) -> float:
    """Calculate a deterministic candidate-job match score."""

    # Required skills = 50 points
    required_score = (
        len(matching_required_skills) / len(required_skills) * 50
        if required_skills
        else 50
    )

    # Preferred skills = 30 points
    preferred_score = (
        len(matching_preferred_skills) / len(preferred_skills) * 30
        if preferred_skills
        else 30
    )

    # Experience = 20 points
    if required_years <= 0:
        experience_score = 20
    else:
        experience_ratio = min(
            candidate_years / required_years,
            1.0,
        )
        experience_score = experience_ratio * 20

    # Total = 50 + 30 + 20 = 100
    score = (
        required_score
        + preferred_score
        + experience_score
    )

    return round(min(score, 100), 2)


def build_crew(
    candidate_profile: str,
    job_positions: str,
) -> Crew:

    profile_analyst = Agent(
        role="Candidate Profile Analyst",
        goal=(
            "Extract and summarize the candidate's key skills, experience level, "
            "and career strengths from their profile."
        ),
        backstory=(
            "You are a meticulous HR analyst who has reviewed thousands of resumes. "
            "You are excellent at pulling concrete skills, years of experience, and "
            "notable achievements out of unstructured candidate profiles."
        ),
        verbose=True,
        llm=MODEL,
    )

    position_analyst = Agent(
        role="Job Requirements Analyst",
        goal=(
            "Break down each job position into its required skills, nice-to-have "
            "skills, and minimum experience level."
        ),
        backstory=(
            "You are a recruiter who specializes in translating job descriptions "
            "into clear, structured requirements that are easy to evaluate against."
        ),
        verbose=True,
        llm=MODEL,
    )

    matchmaker = Agent(
        role="Job Matching Expert",
        goal=(
            "Compare the candidate's profile against each job position and explain "
            "the strengths and gaps of each match."
        ),
        backstory=(
            "You are a senior talent acquisition consultant who carefully compares "
            "candidate evidence with job requirements without inventing information."
        ),
        verbose=True,
        llm=MODEL,
    )

    analyze_candidate = Task(
        description=(
            "Analyze the candidate profile below.\n\n"
            f"{candidate_profile}\n\n"
            "Extract the information accurately. Do not invent information.\n"
            "Return the result in this exact JSON structure:\n\n"
            "{\n"
            '  "skills": ["skill1", "skill2"],\n'
            '  "years_of_experience": 0,\n'
            '  "strengths": ["strength1", "strength2"],\n'
            '  "education": ["education detail"]\n'
            "}"
        ),
        expected_output=(
            "Valid JSON containing skills, years_of_experience, strengths, "
            "and education. Only include information supported by the candidate profile."
        ),
        agent=profile_analyst,
    )

    analyze_positions = Task(
        description=(
            "Analyze the job positions below.\n\n"
            f"{job_positions}\n\n"
            "For each position, identify the required skills, preferred skills, "
            "and minimum experience. Do not invent requirements.\n\n"
            "Return the result in this JSON structure:\n\n"
            "{\n"
            '  "positions": [\n'
            "    {\n"
            '      "title": "Job Title",\n'
            '      "required_skills": ["skill1", "skill2"],\n'
            '      "preferred_skills": ["skill3"],\n'
            '      "minimum_years_experience": 0\n'
            "    }\n"
            "  ]\n"
            "}"
        ),
        expected_output=(
            "Valid JSON containing a list of positions with their titles, "
            "required skills, preferred skills, and minimum experience."
        ),
        agent=position_analyst,
    )

    match_task = Task(
        description=(
            "Using the candidate analysis and job position analysis produced by "
            "the other agents, evaluate the candidate against every position.\n\n"
            "For each position:\n"
            "1. Identify matching required skills.\n"
            "2. Identify missing required skills.\n"
            "3. Identify matching preferred skills.\n"
            "4. Compare the candidate's experience with the required experience.\n"
            "5. Give a short explanation based only on the evidence provided.\n\n"
            "Do not calculate a numerical match score.\n"
            "Do not rank the positions.\n"
            "Do not choose a best-fit position.\n\n"
            "Return the result in this structure:\n\n"
            "{\n"
            '  "matches": [\n'
            "    {\n"
            '      "position": "Job Title",\n'
            '      "matching_required_skills": [],\n'
            '      "missing_required_skills": [],\n'
            '      "matching_preferred_skills": [],\n'
            '      "experience_match": "",\n'
            '      "explanation": ""\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Do not invent candidate skills or job requirements."
        ),
        expected_output=(
            "Valid JSON containing match results for every position, including "
            "matching_required_skills, missing_required_skills, "
            "matching_preferred_skills, experience_match, and explanation. "
            "Do not calculate a numerical score or choose a best-fit position."
        ),
        agent=matchmaker,
        context=[analyze_candidate, analyze_positions],
    )

    return Crew(
        agents=[
            profile_analyst,
            position_analyst,
            matchmaker,
        ],
        tasks=[
            analyze_candidate,
            analyze_positions,
            match_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
