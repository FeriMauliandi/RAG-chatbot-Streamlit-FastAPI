import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader

def load_data(source: str):
    print(f"Mencoba memuat data dari: {source}")
    
    if source.startswith("http://") or source.startswith("https://"):
        loader = WebBaseLoader(source)
        documents = loader.load()
        print("Berhasil memuat data dari Web.")
        
    elif source.lower().endswith(".pdf"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File PDF tidak ditemukan: {source}")
        loader = PyPDFLoader(source)
        documents = loader.load()
        print("Berhasil memuat data dari PDF.")
        
    elif source.lower().endswith(".md"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File Markdown tidak ditemukan: {source}")
        loader = TextLoader(source, encoding="utf-8")
        documents = loader.load()
        print("Berhasil memuat data dari Markdown.")
        
    else:
        raise ValueError("Format tidak didukung. Harap masukkan URL web, file .pdf, atau file .md")
    
    return documents


# == Contoh penggunaan dan testing ==
if __name__ == "__main__":
    test_url = "https://python.langchain.com/docs/get_started/introduction"
    path = "data/hasil.pdf"
    
    try:
        docs = load_data(path)
        print(f"\nJumlah halaman/dokumen yang dimuat: {len(docs)}")
        print(f"Cuplikan isi dokumen pertama:\n{docs[0].page_content[:200]}...\n")
        print(f"Metadata dokumen: {docs[0].metadata}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")