import os
import sys

# Memastikan Python bisa membaca folder 'src'
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from src.ingestion.loader import load_data
from src.ingestion.chunker import split_documents
from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import get_vector_store

# ==========================================
# DAFTAR SEMUA SUMBER DATA
# ==========================================
SOURCES = [
    "data/hasil.pdf",           # PDF      # Markdown
]

def run_ingestion_pipeline():
    print("🚀 Memulai Data Ingestion Pipeline...\n")
    all_chunks = []
    
    # 1. Ekstrak dan Potong semua dokumen
    for source in SOURCES:
        try:
            # Load
            raw_docs = load_data(source)
            # Chunk
            chunks = split_documents(raw_docs, chunk_size=800, chunk_overlap=150)
            all_chunks.extend(chunks)
            print(f"✅ Berhasil memproses: {source} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"❌ Gagal memproses {source}: {e}")
            
    # 2. Masukkan ke Vector Database secara massal
    if all_chunks:
        print(f"\nMenyiapkan model embedding dan menyimpan {len(all_chunks)} chunks ke ChromaDB...")
        embedder = get_embedding_model()
        get_vector_store(chunks=all_chunks, embedding_model=embedder)
        print("\n🔥 Selesai! Semua data telah masuk ke database dan siap digunakan oleh Streamlit.")
    else:
        print("\n⚠️ Tidak ada data yang berhasil diproses.")

if __name__ == "__main__":
    run_ingestion_pipeline()