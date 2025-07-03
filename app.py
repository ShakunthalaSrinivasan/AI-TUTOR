import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from utils import question_mode,quiz_mode,view_quiz_history,plot_score

st.set_page_config(page_title="AI Tutor")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Embeddings + FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vectorstore = FAISS.load_local("vector_store/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Gemini Model

model = genai.GenerativeModel("gemini-2.5-flash")

st.title("AI Tutor for NEET Biology Course")
menu = st.sidebar.radio("Choose an option:",[
                        "Ask a question",
                        "Run Quiz",
                        "View Performance Chart",
                        "Quiz History",
                        "Exit"])

if menu == "Ask a question": 
    question_mode()
elif menu == "Run Quiz":
    quiz_mode()
elif menu == "View Performance Chart":
    plot_score()
elif menu == "Quiz History":
    view_quiz_history()
elif menu == "Exit":
    st.write("Thanks for using AI Tutor!")

if __name__ == "__main__":
    pass