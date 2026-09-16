"""
Run this file to see the crew in action:

    python main.py

Edit CANDIDATE_PROFILE and JOB_POSITIONS below to try your own data.
"""

from crew import build_crew

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

if __name__ == "__main__":
    crew = build_crew(CANDIDATE_PROFILE, JOB_POSITIONS)
    result = crew.kickoff()

    print("\n\n===== FINAL RECOMMENDATION =====\n")
    print(result)
