import sys
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import get_advanced_retriever
from src.generation.prompt import get_rag_prompt

load_dotenv()

# 2. Tangkap API Key dari environment
my_api_key = os.getenv("GOOGLE_API_KEY")

def create_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=os.getenv("GOOGLE_API_KEY")) 
    retriever = get_advanced_retriever(llm=llm, search_type="similarity", final_k=3, fetch_k=5)
    
    prompt = get_rag_prompt()

    # 3. Rangkai menjadi Chain
    def format_docs(docs):
        # ==========================================
        # INTERCEPTOR: Print metadata ke Terminal VS Code
        # ==========================================
        print("\n" + "▼"*50)
        print("🔍 [DEBUG] DOKUMEN YANG DITARIK RETRIEVER:")
        for i, doc in enumerate(docs):
            # Mengambil informasi 'source' dari metadata
            sumber = doc.metadata.get('source', 'Sumber tidak diketahui')
            
            print(f"  [{i+1}] Dari: {sumber}")
            # Jika ingin melihat sedikit cuplikan teks yang ditarik, buka komentar di bawah ini:
            # print(f"      Teks: {doc.page_content[:75]}...") 
        print("▲"*50 + "\n")
        # ==========================================
        
        # Gabungkan teks untuk dikirim ke LLM (kode asli tetap berjalan)
        return "\n\n".join(doc.page_content for doc in docs)
    
    # def llm_no_think(prompt_value):
    #     messages = prompt_value.to_messages()
    #     # Sisipkan /no_think di system message pertama
    #     if messages and messages[0].type == "system":
    #         messages[0].content = "/no_think " + messages[0].content
    #     else:
    #         messages.insert(0, SystemMessage(content="/no_think"))
    #     return llm.invoke(messages)
        
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
