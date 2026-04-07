from langchain_huggingface import HuggingFaceEmbeddings
import torch

model_name = "Qwen/Qwen3-Embedding-0.6B"

def get_embedding_model(model_name: str = model_name):
    print(f"Mempersiapkan model embedding: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware yang digunakan untuk embedding: {device.upper()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} 
    )
    
    return embeddings


# =======Blok testing=========
if __name__ == "__main__":
    embedder = get_embedding_model(model_name)
    
    teks_uji = "Mengubah teks menjadi angka agar dipahami oleh mesin."
    vektor = embedder.embed_query(teks_uji)
    
    print("\n[BERHASIL]")
    print(f"Dimensi vektor (jumlah angka): {len(vektor)}")
    print(f"Cuplikan vektor 5 angka pertama: {vektor[:5]}")