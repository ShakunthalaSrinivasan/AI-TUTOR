import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def load_docs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def embed_and_save(docs, index_path="vector_store/faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS vector store saved to: {index_path}")

if __name__ == "__main__":
    folder = "data"  #folder with PDF files
    all_docs = load_docs(folder)
    embed_and_save(all_docs)
