from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
from PyPDF2 import PdfReader
from googleapiclient.http import MediaIoBaseDownload
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import os
from langchain.document_loaders import GoogleDriveLoader

# Load environment variables from .env file
load_dotenv()

def load_documents():
    CLIENT_SECRET_FILE = os.getenv("CLIENT_SECRET_FILE")
    TOKEN_FILE = os.getenv("TOKEN_FILE")  # Use the correct environment variable here

    loader = GoogleDriveLoader(
        credentials_path=CLIENT_SECRET_FILE,
        token_path=TOKEN_FILE,
        folder_id="1Y66i4ePPEsKeXiGDtXusQQB_NgoFr6IZ",
        recursive=False,
        file_types=["pdf"],
    )
    return loader.load()
    

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

# Load and split the documents
documents = load_documents()
split_documents = split_docs(documents)
print(len(split_documents))

# Continue with the remaining code
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment="northamerica-northeast1-gcp"  # next to the API key in the console
)
index_name = "dole-chatbot"
index = Pinecone.from_documents(split_documents, embeddings, index_name=index_name)

def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

