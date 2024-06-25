import streamlit as st
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from textwrap import wrap
import pandas as pd
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Streamlit interface for API key input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Model selection dropdown
model_option = st.sidebar.selectbox(
    "Choose GPT model",
    ("gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview")
)

# Initialize the OpenAI client
client = None
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.warning("Please enter your OpenAI API key to proceed.")

def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text



def analyze_match(job_description, resume, model):
    if not client:
        st.error("OpenAI client is not initialized. Please enter your API key.")
        return None

    prompt = f"""
    Job Description:
    {job_description}

    Resume:
    {resume}

    Please analyze how well this resume matches the job description. Provide:
    1. A match score from 0 to 100
    2. A brief explanation of the score
    3. Key matching skills or experiences (list exactly 5, use 'N/A' if fewer than 5)
    4. Notable missing qualifications (list exactly 5, use 'N/A' if fewer than 5)
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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert HR assistant skilled in matching resumes to job descriptions. Always provide exactly 5 matching skills and 5 missing qualifications, using 'N/A' if there are fewer than 5."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while analyzing the match: {str(e)}")
        return None

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
            matching_skills = [skill.strip() for skill in skills.split(',') if skill.strip() and skill.strip() != 'N/A']
        elif line.startswith("Missing Qualifications:"):
            missing = line.split(': ')[1] if len(line.split(': ')) > 1 else ""
            missing_qualifications = [qual.strip() for qual in missing.split(',') if qual.strip() and qual.strip() != 'N/A']
        elif line.startswith("Technical Skills:") or line.startswith("Work Experience:") or \
             line.startswith("Education:") or line.startswith("Soft Skills:") or line.startswith("Overall Fit:"):
            criterion, score_str = line.split(': ')
            criteria_scores[criterion] = int(score_str)

    return score, explanation, matching_skills, missing_qualifications, criteria_scores

def create_overall_score_chart(results):
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    candidates = [result[0] for result in sorted_results]
    scores = [result[1] for result in sorted_results]
    wrapped_candidates = ['\n'.join(wrap(name, width=15)) for name in candidates]
    
    fig = go.Figure(data=[
        go.Bar(x=wrapped_candidates, y=scores, text=scores, textposition='auto')
    ])
    
    fig.update_layout(
        title="Overall Candidate Scores",
        xaxis_title="Candidates",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        height=500,
        margin=dict(b=100)
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

# Create the pdf report
def create_pdf_report(results, job_description):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "CV Matcher Analysis Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Job Description: {job_description[:100]}...")

    y_position = height - 120

    for result in results:
        if y_position < 100:  # Start a new page if we're near the bottom
            c.showPage()
            y_position = height - 50

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, f"Candidate: {result[0]}")
        y_position -= 20

        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Score: {result[1]}")
        y_position -= 20

        c.drawString(50, y_position, "Explanation:")
        y_position -= 15
        for line in wrap(result[2], 80):  # Wrap text to fit page width
            c.drawString(70, y_position, line)
            y_position -= 15

        c.drawString(50, y_position, "Matching Skills:")
        y_position -= 15
        for skill in result[3]:
            c.drawString(70, y_position, f"- {skill}")
            y_position -= 15

        c.drawString(50, y_position, "Missing Qualifications:")
        y_position -= 15
        for qual in result[4]:
            c.drawString(70, y_position, f"- {qual}")
            y_position -= 15

        # Add some space before the next candidate
        y_position -= 20

    c.save()
    buffer.seek(0)
    return buffer



def main():
    st.title("CV Matcher - powered by OpenAI ChatGPT")
    st.write("LA - 25/06/2024")
    job_description = st.text_area("Enter the job description:", height=200)
    uploaded_files = st.file_uploader("Upload CV files (PDF only)", type="pdf", accept_multiple_files=True)

    col1, col2 = st.columns(2)
    analyze_button = col1.button("Analyze Matches")
    export_button = col2.button("Export PDF Report")
    
    if st.button("Analyze Matches") and job_description and uploaded_files and client:
        results = []

        for uploaded_file in uploaded_files:
            with pdfplumber.open(uploaded_file) as pdf:
                resume_text = ''
                for page in pdf.pages:
                    resume_text += page.extract_text()

            analysis = analyze_match(job_description, resume_text)
            if analysis:
                print("Raw analysis:", analysis)  # Print raw analysis for debugging
                score, explanation, matching_skills, missing_qualifications, criteria_scores = parse_analysis(analysis)
                results.append((uploaded_file.name, score, explanation, matching_skills, missing_qualifications, criteria_scores))

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        st.sidebar.title("Candidate Rankings")
        ranking_df = pd.DataFrame([(result[0], result[1]) for result in sorted_results], columns=["Candidate", "Score"])
        st.sidebar.dataframe(ranking_df, hide_index=True)

        combined_radar_chart = create_combined_radar_chart(sorted_results)
        st.sidebar.plotly_chart(combined_radar_chart, use_container_width=True)

        overall_score_chart = create_overall_score_chart([(result[0], result[1]) for result in sorted_results])
        st.plotly_chart(overall_score_chart)

        for result in sorted_results:
            st.subheader(f"Candidate: {result[0]}")
            st.write(f"Score: {result[1]}")
            st.write(f"Explanation: {result[2]}")
            st.write("Matching Skills:")
            st.write(", ".join(result[3]))
            st.write("Missing Qualifications:")
            st.write(", ".join(result[4]))

            radar_chart = create_candidate_radar_chart(result[5])
            st.plotly_chart(radar_chart)

            st.markdown("---")

    if export_button and 'results' in st.session_state:
        pdf_buffer = create_pdf_report(st.session_state['results'], job_description)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="cv_matcher_report.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
