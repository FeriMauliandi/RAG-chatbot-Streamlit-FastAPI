import streamlit as st
import os
import sys

# Memastikan Python bisa membaca folder 'src'
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from src.ingestion.loader import load_data
from src.ingestion.chunker import split_documents
from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import get_vector_store
from src.generation.chain import create_rag_chain

# Konfigurasi Halaman
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot RAG Assistant")
st.markdown("Tanyakan apa saja berdasarkan dokumen yang telah diunggah.")

# ==========================================
# CACHING: Agar RAG Chain tidak di-load ulang setiap kali user mengetik
# ==========================================
@st.cache_resource
def load_rag_chain():
    return create_rag_chain()

# ==========================================
# SIDEBAR: Untuk pengaturan Ingestion Data
# ==========================================
with st.sidebar:
    st.header("⚙️ Panel Data")
    
    # Path data real kamu (contoh: komparasi YOLOv11 dan RT-DETR)
    DATA_PATH = "data/hasil.pdf" 
    st.info(f"📁 Target File:\n`{DATA_PATH}`")
    
    if st.button("Proses Dokumen (Ingest)"):
        with st.spinner("Membaca, memotong, dan mengubah teks menjadi vektor..."):
            try:
                raw_docs = load_data(DATA_PATH)
                chunks = split_documents(raw_docs, chunk_size=800, chunk_overlap=150)
                embedder = get_embedding_model()
                get_vector_store(chunks=chunks, embedding_model=embedder)
                st.success("✅ Data berhasil diproses dan masuk ke ChromaDB!")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

# ==========================================
# CHAT INTERFACE
# ==========================================
# Inisialisasi memori chat di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom input untuk user
if prompt := st.chat_input("Tulis pertanyaanmu di sini..."):
    # 1. Tampilkan input user di layar
    with st.chat_message("user"):
        st.markdown(prompt)
    # 2. Simpan input user ke memori
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Proses jawaban dengan AI
    with st.chat_message("assistant"):
        with st.spinner("AI sedang membaca dokumen..."):
            try:
                # Panggil chain RAG
                chain = load_rag_chain()
                jawaban = chain.invoke(prompt)
                
                # Tampilkan jawaban
                st.markdown(jawaban)
                
                # Simpan jawaban ke memori
                st.session_state.messages.append({"role": "assistant", "content": jawaban})
            except Exception as e:
                pesan_error = f"Gagal memproses. Apakah database ChromaDB sudah diisi? Error: {e}"
                st.error(pesan_error)