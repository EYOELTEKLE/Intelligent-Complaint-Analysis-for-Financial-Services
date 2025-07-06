import streamlit as st
from src.rag_pipeline import ComplaintRetriever, PromptEngineer, ComplaintGenerator, RAGPipeline

st.set_page_config(page_title="CrediTrust Complaint Chatbot", layout="wide")
st.title("💬 CrediTrust Complaint Chatbot")
st.markdown("""
Ask questions about customer complaints. The AI will answer using only the most relevant complaint excerpts from the CFPB dataset.\
Sources are shown below each answer for transparency.
""")

# Load RAG pipeline (cache for performance)
@st.cache_resource(show_spinner=True)
def load_rag():
    retriever = ComplaintRetriever(
        index_path="vector_store/faiss.index",
        meta_path="vector_store/metadata.json"
    )
    prompt_engineer = PromptEngineer()
    generator = ComplaintGenerator()
    return RAGPipeline(retriever, prompt_engineer, generator)

rag = load_rag()

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# UI Layout
with st.form(key="chat_form"):
    user_question = st.text_input("Type your question:", key="user_input")
    submitted = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear")

if clear:
    st.session_state.history = []
    st.experimental_rerun()

if submitted and user_question.strip():
    with st.spinner("Retrieving answer..."):
        answer, sources = rag.answer_question(user_question)
        st.session_state.history.append({
            "question": user_question,
            "answer": answer,
            "sources": sources
        })

# Display chat history
for entry in reversed(st.session_state.history):
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**CrediTrust:** {entry['answer']}")
    with st.expander("Show sources", expanded=True):
        for i, src in enumerate(entry["sources"][:2]):
            st.markdown(f"**Source {i+1}:**\n> {src}")
    st.markdown("---")
