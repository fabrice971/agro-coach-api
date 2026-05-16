import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Agro-Coach Mobile", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 🔑 Récupération de la clé (GROQ_API_KEY ou OPENAI_API_KEY)
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

if API_KEY:
    # 🛠️ Astuce : LangChain exige OPENAI_API_KEY, on la force ici
    os.environ["OPENAI_API_KEY"] = API_KEY
    logger.info(f"✅ Clé API configurée (débute par: {API_KEY[:8]}...)")
else:
    logger.error("❌ Aucune clé API trouvée")

llm = None
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="qwen-2.5-32b",
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=800
    )
    logger.info("✅ LLM initialisé avec succès")
except Exception as e:
    logger.error(f"❌ Erreur LLM: {str(e)}")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    location: str = Field(default="Banakorosso")
    user_id: str = Field(default="anonymous")

class QueryResponse(BaseModel):
    status: str
    reponse: str
    conseils: list = None

@app.get("/")
def home():
    return {"service": "Agro-Coach", "status": "running" if llm else "degraded", "key_set": bool(API_KEY)}

@app.get("/health")
def health():
    if llm and API_KEY:
        return {"status": "healthy", "model": "qwen-2.5-32b"}
    return {"status": "degraded", "missing": "API_KEY ou LLM"}

@app.post("/ask")
async def ask(req: QueryRequest):
    if not llm or not API_KEY:
        return {"status": "error", "reponse": "Service indisponible (clé API manquante)", "conseils": []}
    try:
        prompt = f"""Lieu: {req.location}. Question: {req.question}.
Réponds avec EXACTEMENT 3 conseils courts numérotés, moins de 200 mots,
chiffres concrets (FCFA, litres), français simple, contexte Burkina Faso.
Format:
1. [conseil]
2. [conseil]
3. [conseil]"""
        
        resp = llm.invoke(prompt)
        texte = resp.content.strip()
        
        # Extraction robuste
        lignes = [l.strip().lstrip("0123456789.-) ") for l in texte.split('\n') if l.strip() and not l.startswith('```')]
        conseils = lignes[:3] if len(lignes) >= 3 else lignes
        
        return {"status": "success", "reponse": texte, "conseils": conseils}
        
    except Exception as e:
        logger.error(f"❌ Erreur requête: {str(e)}")
        return {"status": "error", "reponse": f"Erreur: {str(e)[:200]}", "conseils": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
