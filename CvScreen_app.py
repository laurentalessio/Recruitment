
import streamlit as st
import os
import tempfile
import PyPDF2
import openai
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from textwrap import wrap
import pandas as pd

# Load environment variables
load_dotenv()



# Add a text input for the OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Ensure the OpenAI API key is set
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please enter your OpenAI API key to proceed.")



def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_match(job_description, resume):
    prompt = f"""
    Job Description:
    {job_description}

    Resume:
    {resume}

    Please analyze how well this resume matches the job description. Provide:
    1. A match score from 0 to 100
    2. A brief explanation of the score
    3. Key matching skills or experiences (list up to 5)
    4. Notable missing qualifications (list up to 5)
    5. Scores for the following criteria (score each from 0 to 100):
       - Technical Skills
       - Work Experience
       - Education
       - Soft Skills
       - Overall Fit

    Format your response exactly as follows:
    Score: [score]
    Explanation: [explanation]
    Matching Skills: [skill1], [skill2], [skill3], [skill4], [skill5]
    Missing Qualifications: [missing1], [missing2], [missing3], [missing4], [missing5]
    Technical Skills: [score]
    Work Experience: [score]
    Education: [score]
    Soft Skills: [score]
    Overall Fit: [score]
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert HR assistant skilled in matching resumes to job descriptions."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def parse_analysis(analysis):
    lines = analysis.split('\n')
    score = 0
    explanation = ""
    matching_skills = []
    missing_qualifications = []
    criteria_scores = {}

    for line in lines:
        if line.startswith("Score:"):
            score = int(line.split(': ')[1])
        elif line.startswith("Explanation:"):
            explanation = line.split(': ')[1]
        elif line.startswith("Matching Skills:"):
            skills = line.split(': ')[1] if len(line.split(': ')) > 1 else ""
            matching_skills = [skill.strip() for skill in skills.split(',') if skill.strip()]
        elif line.startswith("Missing Qualifications:"):
            missing = line.split(': ')[1] if len(line.split(': ')) > 1 else ""
            missing_qualifications = [qual.strip() for qual in missing.split(',') if qual.strip()]
        elif line.startswith("Technical Skills:") or line.startswith("Work Experience:") or \
             line.startswith("Education:") or line.startswith("Soft Skills:") or line.startswith("Overall Fit:"):
            criterion, score_str = line.split(': ')
            criteria_scores[criterion] = int(score_str)

    return score, explanation, matching_skills, missing_qualifications, criteria_scores

def create_overall_score_chart(results):
    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Prepare data
    candidates = [result[0] for result in sorted_results]
    scores = [result[1] for result in sorted_results]
    
    # Wrap candidate names
    wrapped_candidates = ['\n'.join(wrap(name, width=15)) for name in candidates]
    
    fig = go.Figure(data=[
        go.Bar(x=wrapped_candidates, y=scores, text=scores, textposition='auto')
    ])
    
    fig.update_layout(
        title="Overall Candidate Scores",
        xaxis_title="Candidates",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),  # Set y-axis range from 0 to 100
        height=500,  # Increase height to accommodate wrapped text
        margin=dict(b=100)  # Increase bottom margin for x-axis labels
    )
    
    return fig

def create_candidate_radar_chart(criteria_scores):
    categories = list(criteria_scores.keys())
    values = list(criteria_scores.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    return fig

def create_combined_radar_chart(results):
    fig = go.Figure()
    
    categories = list(results[0][5].keys())
    for result in results:
        values = list(result[5].values())
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=result[0]
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Combined Candidate Criteria Scores"
    )
    return fig

def main():
    st.title("CV Matcher")

    # Job description input
    job_description = st.text_area("Enter the job description:", height=200)

    # File uploader for CVs
    uploaded_files = st.file_uploader("Upload CV files (PDF only)", type="pdf", accept_multiple_files=True)

    if st.button("Analyze Matches") and job_description and uploaded_files:
        results = []

        # Process each uploaded CV
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            resume_text = extract_text_from_pdf(temp_file_path)
            os.unlink(temp_file_path)  # Delete the temporary file

            analysis = analyze_match(job_description, resume_text)
            print("Raw analysis:", analysis)  # Print raw analysis for debugging
            score, explanation, matching_skills, missing_qualifications, criteria_scores = parse_analysis(analysis)
            results.append((uploaded_file.name, score, explanation, matching_skills, missing_qualifications, criteria_scores))

        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Sidebar content
        st.sidebar.title("Candidate Rankings")
        ranking_df = pd.DataFrame([(result[0], result[1]) for result in sorted_results], columns=["Candidate", "Score"])
        st.sidebar.dataframe(ranking_df, hide_index=True)

        combined_radar_chart = create_combined_radar_chart(sorted_results)
        st.sidebar.plotly_chart(combined_radar_chart, use_container_width=True)

        # Main content
        overall_score_chart = create_overall_score_chart([(result[0], result[1]) for result in sorted_results])
        st.plotly_chart(overall_score_chart)

        # Display detailed results for each candidate
        for result in sorted_results:
            st.subheader(f"Candidate: {result[0]}")
            st.write(f"Score: {result[1]}")
            st.write(f"Explanation: {result[2]}")
            st.write("Matching Skills:")
            st.write(", ".join(result[3]))
            st.write("Missing Qualifications:")
            st.write(", ".join(result[4]))

            # Create and display radar chart for the candidate
            radar_chart = create_candidate_radar_chart(result[5])
            st.plotly_chart(radar_chart)

            st.markdown("---")  # Add a separator between candidates

if __name__ == "__main__":
    main()
