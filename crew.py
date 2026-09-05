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
            "Analyze the following candidate profile and list their key skills, "
            f"years of experience, and standout strengths:\n\n{candidate_profile}"
        ),
        expected_output=(
            "A structured summary of the candidate's skills, years of experience, "
            "and standout strengths."
        ),
        agent=profile_analyst,
    )

    analyze_positions = Task(
        description=(
            "Analyze the following job positions and break each one down into "
            f"required skills, nice-to-have skills, and experience level:\n\n{job_positions}"
        ),
        expected_output="A structured requirements breakdown for each position.",
        agent=position_analyst,
    )

    match_task = Task(
        description=(
            "Using the candidate summary and the position breakdowns produced by "
            "the other two analysts, determine which position(s) the candidate is "
            "the best fit for. For every position give a match score out of 100 and "
            "a short justification. Finish with a clear recommendation of the single "
            "best-fit position."
        ),
        expected_output=(
            "A ranked list of positions with match scores and reasoning, ending "
            "with one clear recommended position."
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
