import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from datasets import Dataset

# Memastikan Python membaca folder 'src'
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

# Import model lokal milikmu
from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from langchain_ollama import ChatOllama

# Import RAGAS metrics & evaluator
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def build_evaluation_dataset():
    embedder = get_embedding_model()
    vs = get_vector_store(embedding_model=embedder)
    
    test_questions = [
        "apa saja klasifikasi ikan lele",
        "ikan lele berasal dari mana dan kapan masuk ke Indonesia"
    ]
    
    ground_truths = [
        "Kingdom: Animalia, Sub-kingdom: Metazoa, Filum: Chordata, Sub Filum: Vertebrata, Kelas: Pisces, Sub Kelas: Teleostei, Ordo: Ostariophysi, Sub Ordo: Siluroidea, Famili: Clariidae, Genus: Clarias, Spesies: Clarias sp.",
        "Ikan  lele  adalah  salah  satu  ikan  yang  berasal  dari  Taiwan  dan  pertama  kali masuk  ke  Indonesia  pada  tahun  1985  melalui  sebuh  perusahaan  swasta  di  Jakarta"
    ]

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print("🤖 Menjalankan simulasi tanya jawab untuk evaluasi...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=os.getenv("GOOGLE_API_KEY")) 

    for i, question in enumerate(test_questions):
        # A. Proses Retrieval
        results = vs.similarity_search(question, k=3)
        contexts_list = [doc.page_content for doc in results]
        konteks_gabungan = "\n\n".join(contexts_list)
        
        # B. Proses Generation
        prompt = f"Berdasarkan teks ini:\n{konteks_gabungan}\n\nJawab pertanyaan: {question}"
        jawaban_llm = llm.invoke(prompt).content
        
        # C. Simpan ke format RAGAS
        data["question"].append(question)
        data["contexts"].append(contexts_list) # RAGAS butuh format list of strings
        data["answer"].append(jawaban_llm)
        data["ground_truth"].append(ground_truths[i])
        
    # Ubah menjadi HuggingFace Dataset
    return Dataset.from_dict(data)

def run_evaluation():
    print("📊 Menyiapkan Dataset Evaluasi...")
    dataset = build_evaluation_dataset()
    
    # RAGAS membutuhkan akses ke LLM dan Embedder untuk menjadi "Juri"
    juri_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=os.getenv("GOOGLE_API_KEY")) 
    juri_embedder = get_embedding_model()

    print("\n⚖️ Memulai Penilaian RAGAS (Ini akan memakan waktu karena LLM bekerja keras sebagai juri)...")
    
    # Eksekusi RAGAS
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=juri_llm,
        embeddings=juri_embedder
    )
    
    print("\n" + "="*50)
    print("🏆 HASIL EVALUASI RAGAS (Skor 0.0 - 1.0)")
    print("="*50)
    
    # Ubah hasil ke Pandas DataFrame agar rapi
    df_hasil = result.to_pandas()
    
    # Menampilkan rata-rata skor
    print(f"Faithfulness (Anti-Halusinasi) : {df_hasil['faithfulness'].mean():.4f}")
    print(f"Answer Relevancy               : {df_hasil['answer_relevancy'].mean():.4f}")
    print(f"Context Precision              : {df_hasil['context_precision'].mean():.4f}")
    print(f"Context Recall                 : {df_hasil['context_recall'].mean():.4f}")
    
    print("\nDetail per pertanyaan disimpan ke 'eval_report.csv'")
    df_hasil.to_csv("eval_report.csv", index=False)

if __name__ == "__main__":
    run_evaluation()