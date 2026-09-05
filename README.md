# Match Profile to Positions (AI Agent Crew)

An AI-agent version of profile-to-position matching, built with [CrewAI](https://github.com/joaomdmoura/crewAI).

Instead of keyword scoring, three AI agents reason through the match like a hiring team would:

1. **Candidate Profile Analyst** — reads the candidate's profile and summarizes their skills, experience, and strengths
2. **Job Requirements Analyst** — reads the job postings and breaks each one into required skills, nice-to-have skills, and experience level
3. **Job Matching Expert** — compares the two summaries and recommends the best-fit position, with a match score and reasoning for each role

## What you need before running it

- **Python 3.10+** installed on your computer ([python.org/downloads](https://www.python.org/downloads/))
- **An API key** from either [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/settings/keys) (used to power the agents — this project does not include a free key)

## Setup

1. Download/clone this repo, then open a terminal in the project folder.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
4. Open `.env` in a text editor and paste in your API key.
5. Run it:
   ```bash
   python main.py
   ```

You'll see the three agents "think out loud" in the terminal, ending with a final recommendation of the best-fit position for the sample candidate.

## Using your own data

Open `main.py` and edit the `CANDIDATE_PROFILE` and `JOB_POSITIONS` text at the top of the file with your own details, then run it again.

## Project structure

```
match-profile-to-positions/
├── crew.py           # defines the agents and tasks
├── main.py           # entry point — run this file
├── requirements.txt  # Python dependencies
├── .env.example       # template for your API key
└── README.md
```

## Why this project is worth showing

This builds on a simpler rule-based matcher (skills overlap + experience scoring) by replacing the scoring logic with actual AI agents that reason about fit the way a human recruiter would — a good talking point in interviews about the difference between deterministic logic and LLM-based reasoning.
