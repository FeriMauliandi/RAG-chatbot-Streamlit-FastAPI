from langchain_core.prompts import PromptTemplate

def get_rag_prompt():
    """
    Membuat template instruksi ketat untuk LLM.
    """
    template = """Kamu adalah asisten AI yang cerdas. Gunakan potongan konteks berikut untuk menjawab pertanyaan.
jawab dengan tata kata yang menarik, mudah dipahami dan tidak cuek.
Jika jawaban tidak ada di dalam konteks, katakan dengan jujur bahwa kamu tidak tahu. Jangan mengarang jawaban di luar konteks.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )