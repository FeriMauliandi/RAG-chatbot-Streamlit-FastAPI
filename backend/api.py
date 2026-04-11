import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Memastikan Python bisa membaca folder 'src'
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from src.generation.chain import create_rag_chain

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Dictionary global untuk menyimpan model yang dimuat
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Memuat RAG Chain dan ChromaDB ke dalam memori...")
    try:
        # Load chain dan simpan di dictionary global
        ml_models["rag_chain"] = create_rag_chain()
        print("✅ Sistem RAG siap menerima request!")
    except Exception as e:
        print(f"❌ Gagal memuat RAG: {e}")
    
    yield 
    
    ml_models.clear()
    print("🛑 Server dimatikan. Memori dibersihkan.")

# ======fastapi app initialization========
app = FastAPI(
    title="RAG API Engine",
    description="API untuk asisten AI berbasis dokumen",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "RAG API Engine is running. Buka /docs untuk mencoba."}

@app.post("/api/chat", response_model=QueryResponse)
async def chat_with_rag(request: QueryRequest):
    if "rag_chain" not in ml_models:
        raise HTTPException(status_code=503, detail="RAG Model belum siap.")
    
    try:
        # Panggil LLM dan Retriever
        jawaban = ml_models["rag_chain"].invoke(request.question)
        return QueryResponse(answer=jawaban)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")