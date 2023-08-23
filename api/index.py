import os, openai, langchain, pinecone

from fastapi import FastAPI
from langchain.document_loaders import DirectoryLoader, TextLoader,UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from dotenv import load_dotenv

print("loading files")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap  = 0,
    length_function = len,
)

loader = DirectoryLoader('./data', glob="./prod/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

print("splitting files")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print("done splitting files: ", len(texts), "texts")

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME')

app = FastAPI()

@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}

@app.get("/api/env")
def env():
    return {"message": os.getenv("PINECONE_API_KEY")}