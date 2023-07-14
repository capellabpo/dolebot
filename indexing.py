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
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Load PDFs from Google Drive
def load_documents():
    CLIENT_SECRET_FILE = os.getenv("CLIENT_SECRET_FILE") # Use the correct environment variable here
    TOKEN_FILE = os.getenv("TOKEN_FILE")  # Use the correct environment variable here

    loader = GoogleDriveLoader(
        credentials_path=CLIENT_SECRET_FILE,
        token_path=TOKEN_FILE,
        folder_id="1Y66i4ePPEsKeXiGDtXusQQB_NgoFr6IZ",
        recursive=False,
        file_types=["pdf"],
    )
    return loader.load()

# Call the function and get the returned object
documents = load_documents()

print(len(documents))

# Create documents chunks
def split_docs(documents, chunk_size=100, chunk_overlap=20, length_function = len):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

# Load and split the documents
split_documents = split_docs(documents)
print(len(split_documents))

# Create the embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment="asia-southeast1-gcp-free"  # next to the API key in the console
)
index_name = "dole-chatbot2"
index = Pinecone.from_documents(split_documents, embeddings, index_name=index_name)

def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

