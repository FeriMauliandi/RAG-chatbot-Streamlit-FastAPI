from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Menginisialisasi model Sentence Transformer dari HuggingFace.
    Mengembalikan objek HuggingFaceEmbeddings yang siap digunakan.
    """
    print(f"Mempersiapkan model embedding: {model_name}")
    
    # Deteksi otomatis ketersediaan perangkat
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware yang digunakan untuk embedding: {device.upper()}")
    
    # Konfigurasi dan memuat model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} 
    )
    
    return embeddings

# ==========================================
# Blok testing
# ==========================================
if __name__ == "__main__":
    # Inisialisasi model
    embedder = get_embedding_model()
    
    # Mencoba mengubah teks menjadi vektor
    teks_uji = "Mengubah teks menjadi angka agar dipahami oleh mesin."
    vektor = embedder.embed_query(teks_uji)
    
    print("\n[BERHASIL]")
    print(f"Dimensi vektor (jumlah angka): {len(vektor)}")
    print(f"Cuplikan vektor 5 angka pertama: {vektor[:5]}")