from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from meta_ai_api import MetaAI
import requests
import io
import PyPDF2
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

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
    template_document_type: str
    project_document_URL: str
    project_document_type: str


class SkillBasedMatchmakingBody(BaseModel):
    students_skills: List[str]
    project_skills: List[str]


class PredictiveSuccessAnalysisBody(BaseModel):
    students_skills: List[str]
    project_skills: List[str]
    project_description: str
    project_title: str
    number_of_students: int
    student_academic_performance: List[str]
    project_document_URL: str
    project_document_type: str

class QuestionizerBody(BaseModel):
    document_URL: str
    document_type: str
    question: str

class UniqueIdeaDetectionBody(BaseModel):
    project_title: str
    project_abstract_description: str



def download_and_get_text(req):
    r = requests.get(req)
    f = io.BytesIO(r.content)
    reader = PyPDF2.PdfReader(f)
    pages = reader.pages
    text = "".join([page.extract_text() for page in pages])
    return text

def download_and_get_docs(req):
    r = requests.get(req)
    filename = Path('metadata.pdf')
    filename.write_bytes(r.content)
    loader = PyPDFLoader('metadata.pdf')
    return loader.load()


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

    chat_session = model.start_chat(history=[])

    assessment_criteria = chat_session.send_message(assessment_criteria_prompt)
    
    text = download_and_get_text(req.project_document_URL)
    assessment_criteria_prompt = f'''
    STUDENT SUBMISSION:

    {text}

    ASSESSMENT CRITERIA:

    {assessment_criteria.text}

    You are a Automated Project Assessment Tool, provide feedback to STUDENT SUBMISSION using the CRITERIA above.
    '''
        
    chat_session = model.start_chat(history=[])

    assessment_feedback = chat_session.send_message(assessment_criteria_prompt)

    return {"response": assessment_feedback.text}



@app.post("/predictive-success-analysis")
async def index(req: PredictiveSuccessAnalysisBody):
    students_skills = set(req.students_skills)
    project_skills = set(req.project_skills)
    text = download_and_get_text(req.project_document_URL)

    prompt = f'''
    PROJECT DESCRIPTION:
    {req.project_description}
    STUDENT SKILLS:
    {students_skills}
    PROJECT SKILLS:
    {project_skills}
    NUMBER OF STUDENTS:
    {req.number_of_students}
    STUDENT ACADEMIC PERFORMANCE:
    {req.student_academic_performance}
    PROJECT PROPOSAL DOCUMENT:
    {text}

    Based on the above information, predict the success of the project. Suggest ways to enhance their project and what
    skills they require to succeed including tech stack.

    '''
    
    chat_session = model.start_chat(history=[])

    assessment_feedback = chat_session.send_message(prompt)

    return {"response": assessment_feedback.text}


@app.post("/questionizer")
async def index(req: QuestionizerBody):
    text = download_and_get_docs(req.document_URL)

    text_splitter = CharacterTextSplitter(
      separator = "\n",
      chunk_size = 200,
      chunk_overlap = 0
    )

    text_chunks = text_splitter.split_documents(text)
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings(
      model_name=model_name,
      gpt4all_kwargs=gpt4all_kwargs
    )

    db = FAISS.from_documents(text_chunks, embeddings)
    search_results = db.similarity_search(req.question, k=5)
    content = [res.page_content for res in search_results]

    prompt = f'''
    CONTENT:
    {content}

    Based on the CONTENT above,  answer the QUESTION below.
    QUESTION: {req.question}
    '''
    
    res = ai.prompt(prompt)
    return {"response": res['message']}


@app.post("/unique-idea-detection")
async def index(req: UniqueIdeaDetectionBody):
    return req


@app.post("/skill-based-matchmaking")
async def index(req: SkillBasedMatchmakingBody):
    students_skills = set(req.students_skills)
    project_skills = set(req.project_skills)

    required_skills = project_skills - students_skills

    return {"response": required_skills}




