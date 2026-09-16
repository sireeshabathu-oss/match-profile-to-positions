"""
Defines the AI agents and tasks that make up the Match Profile to Positions crew.

Four agents work together in sequence:
1. Profile Analyst  - reads the candidate profile and extracts skills/experience
2. Position Analyst - reads the job postings and extracts requirements
3. Matchmaker        - evaluates the candidate against every position using ONLY
                        stated evidence (no scoring, no ranking, no final pick)
4. Recommender       - reads the Matchmaker's evidence and makes the final call:
                        which position is the best fit, and why
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Which LLM to use. Defaults to a small OpenAI model; swap for an Anthropic
# model (e.g. "anthropic/claude-3-5-sonnet-latest") or Gemini
# (e.g. "gemini/gemini-1.5-flash") if you set the matching API key instead.
MODEL = os.getenv("MODEL", "gpt-4o-mini")


def build_crew(candidate_profile: str, job_positions: str) -> Crew:
    profile_analyst = Agent(
        role="Candidate Profile Analyst",
        goal=(
            "Extract and summarize the candidate's key skills, experience level, "
            "and career strengths from their profile."
        ),
        backstory=(
            "You are a meticulous HR analyst who has reviewed thousands of resumes. "
            "You are excellent at pulling out concrete skills, years of experience, "
            "and notable achievements from unstructured candidate profiles."
        ),
        verbose=True,
        llm=MODEL,
    )

    position_analyst = Agent(
        role="Job Requirements Analyst",
        goal=(
            "Break down each job position into its required skills, nice-to-have "
            "skills, and experience level."
        ),
        backstory=(
            "You are a recruiter who specializes in translating vague job "
            "descriptions into clear, structured requirements."
        ),
        verbose=True,
        llm=MODEL,
    )

    matchmaker = Agent(
        role="Job Matching Expert",
        goal=(
            "Evaluate the candidate against every job position strictly using "
            "stated evidence — no scoring, no ranking, no final pick."
        ),
        backstory=(
            "You are a careful evaluator who reports only what the evidence "
            "shows. You never invent skills or requirements, and you never "
            "jump to a conclusion about which role is 'best' — that decision "
            "belongs to someone else."
        ),
        verbose=True,
        llm=MODEL,
    )

    recommender = Agent(
        role="Hiring Recommendation Lead",
        goal=(
            "Read the Matchmaker's evidence-based evaluation and make the final "
            "call: which position is the best fit for this candidate, and why."
        ),
        backstory=(
            "You are a senior talent acquisition lead who makes the final call "
            "on candidate-role fit. You base your recommendation strictly on "
            "the evidence already gathered — matching skills, missing skills, "
            "and experience fit — and explain your reasoning clearly."
        ),
        verbose=True,
        llm=MODEL,
    )

    analyze_candidate = Task(
        description=(
            "Analyze the following candidate profile and list their key skills, "
            f"experience, and strengths:\n\n{candidate_profile}"
        ),
        expected_output=(
            "A structured summary of the candidate's skills, years of "
            "experience, and standout strengths."
        ),
        agent=profile_analyst,
    )

    analyze_positions = Task(
        description=(
            "Analyze the following job positions and break each one down into "
            f"required skills, nice-to-have skills, and experience level:\n\n{job_positions}"
        ),
        expected_output="A structured breakdown of each position's requirements.",
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

    recommend_task = Task(
        description=(
            "Using ONLY the Matchmaker's evidence (matching/missing required "
            "skills, matching preferred skills, and experience comparison for "
            "every position), decide which position is the best fit for this "
            "candidate.\n\n"
            "Base your decision strictly on:\n"
            "- How many required skills are met vs. missing\n"
            "- How many preferred skills are met\n"
            "- Whether experience meets or exceeds the requirement\n\n"
            "Do not introduce any new information not already present in the "
            "Matchmaker's evidence. If two positions are genuinely tied on the "
            "evidence, say so explicitly instead of guessing.\n\n"
            "Return the result in this structure:\n\n"
            "{\n"
            '  "recommended_position": "Job Title or null if tied/unclear",\n'
            '  "reasoning": "Explanation grounded in the evidence above",\n'
            '  "runner_up": "Job Title or null if not applicable"\n'
            "}"
        ),
        expected_output=(
            "Valid JSON with recommended_position, reasoning, and runner_up, "
            "based strictly on the Matchmaker's evidence."
        ),
        agent=recommender,
        context=[match_task],
    )

    return Crew(
        agents=[profile_analyst, position_analyst, matchmaker, recommender],
        tasks=[analyze_candidate, analyze_positions, match_task, recommend_task],
        process=Process.sequential,
        verbose=True,
    )
