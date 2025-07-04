import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils import quiz_mode, question_mode, plot_score, view_quiz_history

# --- Streamlit Config ---
st.set_page_config(page_title="AI Tutor")

st.title("AI Tutor for NEET Biology")

# Load .env variables
load_dotenv()

# Configure API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Load Vectorstore ---

@st.cache_resource
def load_vectorstore(index_path="vector_store/faiss_index"):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

retriever = load_vectorstore()

# --- Load Gemini Model ---
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Sidebar Menu ---
menu = st.sidebar.radio("Choose an option:", [
    "Ask a question",
    "Run Quiz",
    "View Performance Chart",
    "Quiz History",
    "Exit"
])

# --- Routing Logic ---
if menu == "Ask a question":
    question_mode(retriever, model)

elif menu == "Run Quiz":
    quiz_mode(retriever, model)

elif menu == "View Performance Chart":
    plot_score()

elif menu == "Quiz History":
    view_quiz_history()

elif menu == "Exit":
    st.write("Thanks for using AI Tutor!")
