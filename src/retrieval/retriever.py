import os
import sys  

def get_retriever(vector_store, search_type="similarity", k=4):
    """
    Mengubah Vector Store menjadi objek Retriever untuk mencari dokumen relevan.
    
    Args:
        vector_store: Objek ChromaDB yang sudah dimuat.
        search_type: Tipe pencarian ('similarity' atau 'mmr').
        k: Jumlah potongan dokumen (chunks) teratas yang ingin diambil.
    """
    print(f"Menyiapkan Retriever (Tipe: {search_type}, Mengambil {k} dokumen)")
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
    return retriever

# ==========================================
# Blok testing
# ==========================================
if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(root_dir)

    from src.ingestion.embedder import get_embedding_model
    from src.retrieval.vector_store import get_vector_store
    
    embedder = get_embedding_model()
    vs = get_vector_store(embedding_model=embedder)
    
    retriever = get_retriever(vs, k=1)
    
    pertanyaan = "Apa itu ChromaDB?"
    print(f"\nPertanyaan: '{pertanyaan}'")
    
    hasil_dengan_skor = vs.similarity_search_with_score(pertanyaan, k=3)
    
    print("\nHasil Pencarian dan Skor Jarak (Makin kecil angkanya, makin relevan):")
    for doc, skor in hasil_dengan_skor:
        print(f"Skor: {skor:.4f} | Teks: {doc.page_content}")
    
    hasil_pencarian = retriever.invoke(pertanyaan)
    
    print("\nHasil Pencarian Paling Relevan:")
    for i, doc in enumerate(hasil_pencarian):
        print(f"{i+1}. {doc.page_content}")