import streamlit as st
import os
import sys

# Memastikan Python bisa membaca folder 'src'
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

# Kita HANYA mengimpor fungsi untuk memuat RAG Chain
from src.generation.chain import create_rag_chain

# Konfigurasi UI Streamlit
st.set_page_config(page_title="AI Engineer Copilot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Assistant (Production Mode)")
st.markdown("Database telah dimuat. Silakan ajukan pertanyaan seputar dokumenmu.")

# CACHING: Memuat RAG Chain (dan ChromaDB) satu kali saja
@st.cache_resource
def load_rag_chain():
    # Fungsi ini di dalam chain.py akan otomatis memanggil Chroma()
    # dan membaca folder ./chroma_db tanpa melakukan ingestion baru
    return create_rag_chain()

# ==========================================
# CHAT INTERFACE
# ==========================================
# Inisialisasi memori chat di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat sebelumnya agar tidak hilang saat halaman direfresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom input untuk user
if prompt := st.chat_input("Tulis pertanyaan spesifik mengenai dokumen..."):
    # 1. Tampilkan pertanyaan user di layar
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Simpan ke memori Streamlit
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Proses RAG dan LLM
    with st.chat_message("assistant"):
        with st.spinner("Mencari di ChromaDB dan memformulasikan jawaban..."):
            try:
                # Memanggil chain yang sudah di-cache
                chain = load_rag_chain()
                
                # Eksekusi pencarian & generasi jawaban
                jawaban = chain.invoke(prompt)
                
                # Tampilkan jawaban dari LLM
                st.markdown(jawaban)
                
                # Simpan jawaban ke memori
                st.session_state.messages.append({"role": "assistant", "content": jawaban})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan sistem: {e}\n\nPastikan folder 'chroma_db' sudah ada isinya.")