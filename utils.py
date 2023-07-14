from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from dotenv import load_dotenv
import os
import requests


# Load environment variables from .env file
load_dotenv()

# Provide your OpenAI API key here
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

model = SentenceTransformer('all-MiniLM-L6-v2')


# Provide your Pinecone API key here
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key, environment='asia-southeast1-gcp-free')
index = pinecone.Index('dole-chatbot2')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)

    if len(result['matches']) >= 2:
        match1 = result['matches'][0]['metadata']['text']
        match2 = result['matches'][1]['metadata']['text']
        return match1 + "\n" + match2
    elif len(result['matches']) == 1:
        match1 = result['matches'][0]['metadata']['text']
        return match1
    else:
        return "No matches found"

def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=100, 
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']      

                                                                                                                                                                                                                                                         
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i] + "\n"
    return conversation_string

def redirect_to_openai(query):
    base_url = "https://chat.openai.com/api/create"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "text-davinci-003",
        "prompt": query,
        "temperature": 0.7,
        "max_tokens": 60,
        "stop": "\n"
    }
    response = requests.post(base_url, headers=headers, json=data)
    return response.json()["choices"][0]["text"]




