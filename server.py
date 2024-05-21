from fastapi import FastAPI
from pydantic import BaseModel
from meta_ai_api import MetaAI
import requests
import io
import PyPDF2
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 0,
  "max_output_tokens": 2048,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.0-pro",
  safety_settings=safety_settings,
  generation_config=generation_config,
)

class ProjectAssessmentBody(BaseModel):
    template_document_URL: str
    project_document_URL: str


def download_and_get_text(req):
    r = requests.get(req)
    f = io.BytesIO(r.content)
    reader = PyPDF2.PdfReader(f)
    pages = reader.pages
    text = "".join([page.extract_text() for page in pages])
    return text


app = FastAPI()
ai = MetaAI()

@app.post("/automated-project-assessment")
async def index(req: ProjectAssessmentBody):

    text = download_and_get_text(req.template_document_URL)

    assessment_criteria_prompt = f'''
    TEMPLATE:

    {text}
    
    You are a Automated Project Assessment Tool, create a summarized document from the above TEMPLATE that contains main points and criteria to assess a student's submission. Do not add points or marks.
    '''
    assessment_criteria = ai.prompt(assessment_criteria_prompt)
    
    text = download_and_get_text(req.project_document_URL)
    assessment_criteria_prompt = f'''
    STUDENT SUBMISSION:

    {text}

    ASSESSMENT CRITERIA:

    {assessment_criteria['message']}

    You are a Automated Project Assessment Tool, provide feedback to STUDENT SUBMISSION using the CRITERIA above.
    '''

        
    chat_session = model.start_chat(history=[])

    assessment_feedback = chat_session.send_message(assessment_criteria_prompt)

    return {"response": assessment_feedback.text}




@app.post("/predictive-success-analysis")
async def index(req: ProjectAssessmentBody):
    return req


@app.post("/questionizer")
async def index(req: ProjectAssessmentBody):
    return req



