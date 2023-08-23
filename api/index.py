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

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME')

pinecone.init(
        api_key = PINECONE_API_KEY,
        environment = PINECONE_ENV
)

index = pinecone.Index(INDEX_NAME)
index.describe_index_stats()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

print("done loading embeddings: ", len(embeddings), "embeddings")

docsearch = Pinecone.from_documents(texts, embeddings, index_name = INDEX_NAME)

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def parse_response(response):
    print(response['result'])
    print('\n\nSources:')
    for source_name in response["source_documents"]:
        print(source_name.metadata['source'], "page #:", source_name.metadata['page'])

retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

app = FastAPI()

@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}

@app.get("/api/env")
def env():
    return {"message": os.getenv("PINECONE_API_KEY")}