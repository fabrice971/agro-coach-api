import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Agro-Coach Mobile API", version="1.0.0")

# 🔓 CORS obligatoire pour que Gradio (HF) puisse appeler Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚙️ Configuration Groq + Qwen
llm = ChatOpenAI(
    model="qwen-2.5-32b",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.1,
    max_tokens=800
)

# 🤖 Agent unique optimisé mobile
agent = Agent(
    role="Assistant Agro Mobile Burkina",
    goal="Fournir des conseils agricoles courts, pratiques et actionnables pour smartphone",
    backstory="Expert en agriculture sahélienne. Vous répondez en français simple, avec des listes courtes, des chiffres concrets (FCFA, litres, heures), et des conseils testés au Burkina Faso.",
    llm=llm,
    verbose=False
)

class QueryRequest(BaseModel):
    question: str
    location: str = "Banakorosso"
    user_id: str = "anonymous"

class QueryResponse(BaseModel):
    status: str
    reponse: str
    conseils: list[str] | None = None

@app.get("/")
def home():
    return {"service": "Agro-Coach Mobile API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "qwen-2.5-32b via Groq"}

@app.post("/ask", response_model=QueryResponse)
async def ask(query: QueryRequest):
    try:
        task = Task(
            description=f"""
            Utilisateur: {query.user_id} | Lieu: {query.location}
            Question: {query.question}
            
            Instructions strictes:
            - Réponds en MAXIMUM 3 points courts (moins de 200 mots total)
            - Utilise des chiffres concrets (FCFA, litres, heures, jours)
            - Format mobile : listes numérotées, pas de paragraphes longs
            - Langue : français simple, accessible
            - Contexte : climat sahélien, ressources limitées
            """,
            expected_output="1) Conseil principal 2) Étapes pratiques 3) Précautions",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        raw_text = result.raw.strip()
        
        # Extraction propre des 3 conseils
        lines = [l.strip().lstrip("0123456789.-) ") for l in raw_text.split('\n') if l.strip()]
        conseils = lines[:3] if len(lines) >= 3 else lines
        
        return QueryResponse(
            status="success",
            reponse=raw_text,
            conseils=conseils
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API: {str(e)}")
