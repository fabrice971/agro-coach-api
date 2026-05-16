import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Logging pour voir les erreurs dans Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Agro-Coach Mobile", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 Vérification de la clé API AU DÉMARRAGE
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    logger.error("❌ GROQ_API_KEY non trouvée dans les variables d'environnement")
    logger.error("💡 Ajoute-la dans Render : Dashboard → Environment Variables")
else:
    logger.info(f"✅ GROQ_API_KEY trouvée (commence par: {GROQ_KEY[:8]}...)")

# Initialisation LLM avec try/except
llm = None
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="qwen-2.5-32b",
        openai_api_key=GROQ_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=800
    )
    logger.info("✅ LLM initialisé avec succès")
except Exception as e:
    logger.error(f"❌ Erreur initialisation LLM: {str(e)}")

# Modèles
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    location: str = Field(default="Banakorosso")
    user_id: str = Field(default="anonymous")

class QueryResponse(BaseModel):
    status: str
    reponse: str
    conseils: list = None

# Endpoints
@app.get("/")
def home():
    return {
        "service": "Agro-Coach Mobile API",
        "status": "running" if llm else "degraded (LLM not ready)",
        "groq_key_set": bool(GROQ_KEY)
    }

@app.get("/health")
def health():
    if llm and GROQ_KEY:
        return {"status": "healthy", "model": "qwen-2.5-32b"}
    return {"status": "degraded", "missing": "GROQ_API_KEY" if not GROQ_KEY else "LLM init"}

@app.post("/ask")
async def ask(req: QueryRequest):
    # Vérification avant traitement
    if not llm or not GROQ_KEY:
        logger.warning(f"Requête reçue mais LLM non prêt : {req.question[:50]}...")
        return {
            "status": "error",
            "reponse": "Service temporairement indisponible. Vérifiez la configuration GROQ_API_KEY.",
            "conseils": []
        }
    
    try:
        logger.info(f"📥 Requête: user={req.user_id}, lieu={req.location}, question={req.question[:100]}")
        
        prompt = f"""Lieu: {req.location}. Question: {req.question}.
Réponds avec EXACTEMENT 3 conseils courts numérotés, moins de 200 mots,
chiffres concrets (FCFA, litres), français simple, contexte Burkina Faso.
Format:
1. [conseil]
2. [conseil]
3. [conseil]"""
        
        resp = llm.invoke(prompt)
        texte = resp.content.strip()
        logger.info(f"📤 Réponse générée ({len(texte)} chars)")
        
        # Extraction robuste
        lignes = []
        for l in texte.split('\n'):
            l = l.strip()
            if l and not l.startswith('```'):
                for prefix in ['1.','2.','3.','1)','2)','3)','-','*']:
                    if l.startswith(prefix):
                        l = l[len(prefix):].strip()
                        break
                if l:
                    lignes.append(l)
        
        conseils = lignes[:3] if len(lignes) >= 3 else lignes
        
        return {
            "status": "success",
            "reponse": texte,
            "conseils": conseils
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement requête: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "reponse": f"Erreur interne: {str(e)[:200]}",
            "conseils": []
        }

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Démarrage sur le port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
