# Open Source HuggingFace Text Embedding Model
# Chroma DB for Storage

#all-mpnet-base-v2 -> This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional 
# dense vector space and can be used for tasks like clustering or semantic search.

#We will use Mistral Open Source Model by Hugging face API.

import langchain
import huggingface_hub
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
import chromadb
import os

#--------Embedding Library---------
from langchain.embeddings import HuggingFaceEmbeddings

# ----------PDF Loader----------
from langchain.document_loaders import PyPDFLoader

#-----. Text Splitter Library .-------
from langchain.text_splitter import RecursiveCharacterTextSplitter

#--------ChromaDb---------------
from langchain.vectorstores import Chroma
#--------------------------------


model = "mistralai/Mistral-7B-Instruct-v0.3" # Open Source LLM
hf_key = "<hf-API KEY>" # Hugging-Face API
path = "numerical_analysis.pdf" # Your pdf file path, example is given


#----------PDF Reader-----------------

loader = PyPDFLoader(path)
docs = loader.load()

# -----------Text Splitter ---------
text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 1500,
chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name = model_name) 

# ----------Embedding------------------
persist_dir = os.getcwd() # Saving the Vector embedding
vectordb = Chroma.from_documents(documents = splits,
                                 embedding = hf,
                                 persist_directory = persist_dir)    

#-----------------------------------------------------------   
llm  = HuggingFaceEndpoint(repo_id = model,
                           max_length = 100,
                           temperature = 0.1,
                           token = hf_key)

memory = ConversationBufferMemory(
memory_key="chat_history",
return_messages=True
)
retriever=vectordb.as_retriever()

qa_r = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    output_key = 'answer',
    memory=memory
    )

# -------Enter "exit" to stop the conversation-----------
answering  = True
while answering:

  que = input("Question: ")
  result = qa_r({f"question":que})

  print(result['answer'])

  if que.lower() == "exit":
    answering = False
