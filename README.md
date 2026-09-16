# Match Profile to Positions

An AI-agent system that analyzes a candidate profile and job positions, identifies skill matches and gaps, and calculates a transparent match score using deterministic Python logic.

The project uses [CrewAI](https://github.com/crewAIInc/crewAI) to handle the language-understanding part of the process, while Python handles the final scoring.

## Overview

The system takes:

* A candidate profile
* One or more job positions

It then:

1. Extracts the candidate's skills, experience, strengths, and education using an LLM agent.
2. Extracts the required skills, preferred skills, and minimum experience for each position using another LLM agent.
3. Compares the candidate's skills with each position's requirements.
4. Calculates a deterministic match score using Python.
5. Displays matching skills, missing required skills, preferred-skill matches, experience information, and the final score.

## Architecture

The project uses three AI agents:

### 1. Candidate Profile Analyst

Analyzes the candidate profile and extracts structured information:

* Skills
* Years of experience
* Strengths
* Education

The agent is instructed to return the information as JSON.

### 2. Job Requirements Analyst

Analyzes the job positions and extracts:

* Required skills
* Preferred skills
* Minimum years of experience
* Position title

The agent also returns structured JSON.

### 3. Job Matching Expert

Compares the candidate analysis with the job requirements and identifies:

* Matching required skills
* Missing required skills
* Matching preferred skills
* Experience match
* Explanation of the match

The final numerical score is **not calculated by the LLM**.

## Deterministic Scoring

The final score is calculated in Python using a fixed weighting system.

| Category         |         Weight |
| ---------------- | -------------: |
| Required skills  |      50 points |
| Preferred skills |      30 points |
| Experience       |      20 points |
| **Total**        | **100 points** |

### Required Skills

The candidate receives up to 50 points based on the percentage of required skills they have.

```text
required skill score =
matching required skills / total required skills × 50
```

### Preferred Skills

The candidate receives up to 30 points based on the percentage of preferred skills they have.

```text
preferred skill score =
matching preferred skills / total preferred skills × 30
```

### Experience

The candidate receives up to 20 points based on their experience compared with the required experience.

The experience contribution is capped at 20 points.

```text
experience ratio =
candidate experience / required experience
```

The ratio is capped at `1.0`, so exceeding the required experience does not produce more than 20 points.

### Final Score

```text
final score =
required skill score
+ preferred skill score
+ experience score
```

The final result is capped at 100.

## Why Use LLMs and Deterministic Logic?

The project intentionally separates language understanding from numerical calculation.

The LLM is useful for interpreting unstructured text such as:

* Candidate profiles
* Job descriptions
* Skill descriptions
* Experience statements

Python is used for the numerical calculation because the scoring rules should be:

* Transparent
* Reproducible
* Easy to test
* Independent of LLM variability

This hybrid approach allows the project to use LLMs where language understanding is useful while keeping the final scoring logic deterministic.

## Example

The sample candidate is:

```text
Name: Priya Nair
Experience: 5 years
Skills: Python, SQL, Machine Learning, Data Visualization, A/B Testing
```

The sample positions include:

```text
Senior Data Scientist
Required skills: Python, Machine Learning, SQL
Nice-to-have skills: Data Visualization, A/B Testing
Minimum experience: 4 years
```

and:

```text
Frontend Engineer
Required skills: JavaScript, React
Nice-to-have skills: CSS, TypeScript
Minimum experience: 1 year
```

The resulting scores are:

```text
Senior Data Scientist
Match Score: 100.0/100

Frontend Engineer
Match Score: 20.0/100
```

The output also shows which required skills match and which required skills are missing.

## Project Structure

```text
match-profile-to-positions/
│
├── crew.py
│   ├── Defines the AI agents
│   ├── Defines the AI tasks
│   └── Contains the deterministic scoring function
│
├── main.py
│   ├── Contains sample candidate and job data
│   ├── Runs the CrewAI workflow
│   ├── Parses structured agent output
│   ├── Calculates final match results
│   └── Displays the results
│
├── requirements.txt
│   └── Python dependencies
│
├── .env.example
│   └── Environment variable template
│
├── .gitignore
│
└── LICENSE
```

## Requirements

* Python 3.10+
* A compatible LLM API key
* CrewAI
* python-dotenv

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sireeshabathu-oss/match-profile-to-positions.git
cd match-profile-to-positions
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Copy the environment template:

```bash
cp .env.example .env
```

Then open `.env` and add your API key.

The project reads the model configuration from the environment.

By default, the code uses:

```text
MODEL=gpt-4o-mini
```

if no other model is specified.

### 4. Run the project

```bash
python main.py
```

The CrewAI agents will run sequentially, after which Python calculates and displays the final match results.

## Using Your Own Candidate and Job Data

Open `main.py` and modify:

```python
CANDIDATE_PROFILE = """
Your candidate profile here
"""
```

and:

```python
JOB_POSITIONS = """
Your job positions here
"""
```

Then run:

```bash
python main.py
```

The project will analyze the new information and calculate the match scores.

## Design Decisions

### LLM for extraction

Candidate profiles and job descriptions can contain unstructured natural language. LLM agents are used to convert this information into structured data.

### Python for scoring

The numerical score is calculated independently from the LLM output.

This makes the scoring formula explicit and reproducible.

### Sequential agent workflow

The agents run in sequence:

```text
Candidate Profile
       │
       ▼
Profile Analyst
       │
       ▼
Structured Candidate Data
       
Job Positions
       │
       ▼
Requirements Analyst
       │
       ▼
Structured Job Data
       
       └──────────────┐
                      ▼
                Matchmaker
                      │
                      ▼
              Match Analysis
                      │
                      ▼
              Python Scoring
                      │
                      ▼
              Final Results
```

## Current Limitations

This is a portfolio project and is not intended to be a production hiring system.

Current limitations include:

* Skill matching is based primarily on normalized skill names.
* Synonyms and closely related skills may not always be recognized as equivalent.
* The scoring weights are manually defined.
* Input data is currently provided directly in `main.py`.
* There is currently no web interface.
* The system does not ingest resume files such as PDF or DOCX.
* The scoring model can be improved further, particularly around missing required skills.
* The project currently uses a small sample dataset rather than a formal evaluation dataset.

## Future Improvements

Possible next improvements include:

* Automated unit tests for the scoring logic
* Improved skill normalization and synonym handling
* Better handling of missing required skills
* Resume PDF/DOCX input
* A larger evaluation dataset
* Structured output validation
* A web interface
* More detailed candidate skill-gap analysis
* Evaluation of scoring consistency across different candidate profiles and positions

## Tech Stack

* **Python** — application and scoring logic
* **CrewAI** — multi-agent orchestration
* **LLM** — natural-language extraction and reasoning
* **python-dotenv** — environment configuration
* **Git / GitHub** — version control

## Project Status

**Working prototype**

The current version demonstrates a hybrid AI approach:

> LLMs interpret the candidate and job descriptions; Python calculates the final match score.

The project is being developed incrementally with a focus on transparent scoring, reproducibility, and practical AI-agent architecture.

