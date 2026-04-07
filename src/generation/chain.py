import sys
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import get_retriever
from src.generation.prompt import get_rag_prompt

def create_rag_chain():
    embedder = get_embedding_model()
    vs = get_vector_store(embedding_model=embedder)
    retriever = get_retriever(vs, k=5) 
    
    prompt = get_rag_prompt()
    
    llm = ChatOllama(model="qwen3.5:cloud", temperature=0.4) 
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# =======Blok testing=========
if __name__ == "__main__":
    chain = create_rag_chain()
    
    pertanyaan = "Apa itu LangChain. chromaDB dan pinecone?"
    print(f"\nUser: {pertanyaan}")
    print("AI sedang berpikir (memproses via LLM lokal)...\n")
    
    jawaban = chain.invoke(pertanyaan)
    
    print("=== JAWABAN RAG ===")
    print(jawaban)