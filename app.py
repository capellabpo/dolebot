from dotenv import load_dotenv
import streamlit as st
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import tiktoken
import json
import os

def main():
    load_dotenv()
    st.set_page_config(
        page_title="dolebot",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .header {
            color: #3366ff;
            font-size: 36px;
            padding: 20px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h1 class='header'>Ask your DOLE BOT 💬</h1>", unsafe_allow_html=True)
    
    # Authenticate and create Google Drive service
    credentials_json_path = os.getenv("GOOGLE_CREDENTIALS_JSON_PATH")
    with open(credentials_json_path) as f:
        service_account_info = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    drive_service = build('drive', 'v3', credentials=credentials)


    # Upload file from Google Drive
    drive_file_id = "14U7paDJntd7fOW6WPW2XwXV7NLy1lP8s"  # Replace with the actual Google Drive file ID
    if drive_file_id:
        try:
            request = drive_service.files().get_media(fileId=drive_file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            pdf_reader = PdfReader(fh)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)

                st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()

  
