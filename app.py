import os
import streamlit as st
import google.generativeai as genai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils import quiz_mode, question_mode, plot_score, view_my_results

# Streamlit Config
st.set_page_config(page_title="AI Tutor")

st.title("AI Tutor for NEET Biology")

# Load API keys from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Configure APIs 
genai.configure(api_key=gemini_api_key)

# Load Vectorstore 

@st.cache_resource
def load_vectorstore(index_path="vector_store/faiss_index"):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True
                                  )
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
    "View my Results",
    "Exit"
])

# --- Routing Logic ---
if menu == "Ask a question":
    question_mode(retriever, model)

elif menu == "Run Quiz":
    quiz_mode(retriever, model)

elif menu == "View Performance Chart":
    plot_score()

elif menu == "View my Results":
    view_my_results()

elif menu == "Exit":
    st.write("Thanks for using AI Tutor!")
