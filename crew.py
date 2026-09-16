"""
Defines the AI agents and tasks that make up the Match Profile to Positions crew.

Three agents work together in sequence:
1. Profile Analyst  - reads the candidate profile and extracts skills/experience
2. Position Analyst - reads the job postings and extracts requirements
3. Matchmaker        - compares the two and recommends the best-fit role(s)
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Which LLM to use. Defaults to a small OpenAI model; swap for an Anthropic
# model (e.g. "anthropic/claude-3-5-sonnet-latest") if you set ANTHROPIC_API_KEY
# instead of OPENAI_API_KEY in your .env file.
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
            "Compare the candidate's profile against each job position and "
            "recommend the best fit, with clear reasoning."
        ),
        backstory=(
            "You are a senior talent acquisition consultant known for making sharp, "
            "well-justified matches between candidates and open roles."
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
        "5. Calculate a match score from 0 to 100 based only on the evidence provided.\n"
        "6. Give a short explanation for the score.\n\n"
        "Return the result in this structure:\n\n"
        "{\n"
        '  "matches": [\n'
        "    {\n"
        '      "position": "Job Title",\n'
        '      "match_score": 0,\n'
        '      "matching_skills": [],\n'
        '      "missing_skills": [],\n'
        '      "experience_match": "",\n'
        '      "explanation": ""\n'
        "    }\n"
        "  ],\n"
        '  "best_fit": "Job Title"\n'
        "}\n\n"
        "Do not invent candidate skills or job requirements."
    ),
    expected_output=(
        "Valid JSON containing match results for every position, including "
        "match_score, matching_skills, missing_skills, experience_match, "
        "explanation, and best_fit."
    ),
    agent=matchmaker,
    context=[analyze_candidate, analyze_positions],
)

    return Crew(
        agents=[profile_analyst, position_analyst, matchmaker],
        tasks=[analyze_candidate, analyze_positions, match_task],
        process=Process.sequential,
        verbose=True,
    )
