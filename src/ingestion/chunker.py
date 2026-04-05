from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

chunk_size = 2000
chunk_overlap = 200

#   documents: List dokumen dari loader.py
#   chunk_size: Jumlah karakter maksimal per potongan.
#   chunk_overlap: Jumlah karakter yang tumpang tindih antar potongan.

def split_documents(documents: List[Document], chunk_size, chunk_overlap):
    """
    Memotong daftar dokumen menjadi bagian-bagian kecil (chunks).
    
    """
    print(f"Memulai proses pemotongan {len(documents)} dokumen...")
    
    # Inisialisasi splitter
    # Kita menggunakan karakter [\n\n, \n, " ", ""] sebagai prioritas pemisah
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Berhasil memecah dokumen menjadi {len(chunks)} potongan teks.")
    return chunks

# ==========================================
# Blok testing
# ==========================================
if __name__ == "__main__":
    # Contoh penggunaan sederhana
    from loader import load_data
    
    url = "https://python.langchain.com/docs/introduction/"
    path = "data/hasil.pdf"
    
    try:
        raw_docs = load_data(path)
        final_chunks = split_documents(raw_docs, chunk_size, chunk_overlap)
        
        if len(final_chunks) > 1:
            print(f"\nIsi Chunk #1:\n{final_chunks[0].page_content[:150]}...")
            print(f"\nIsi Chunk #2:\n{final_chunks[1].page_content[:150]}...")
            print(f"\nMetadata Chunk #1: {final_chunks[0].metadata}")
    except Exception as e:
        print(f"Error saat testing chunker: {e}")