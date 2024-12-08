

import nest_asyncio

from pydantic import BaseModel
import faiss
from langchain.chains import ConversationalRetrievalChain
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
from uuid import uuid4
from sentence_transformers import SentenceTransformer
import faulthandler
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts.prompt import PromptTemplate

from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn
import requests

directory_path = "example_data/"

loader = PyPDFDirectoryLoader(directory_path)
documents = loader.load()

if not documents:
    raise ValueError("No documents loaded. Check the directory path or file format.")
for i, doc in enumerate(documents):
    if not doc.page_content.strip():
        print(f"Document {i} is empty.")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(documents)
#print ("len(docs)", len(docs))
print ("docs[0]", docs[0])

embed_model_id = "sentence-transformers/all-mpnet-base-v2"
local_model_dir = "./embedding_model"

model = SentenceTransformer(embed_model_id)
local_model_dir = "./embedding_model"
model.save(local_model_dir)
print(f"Model saved locally at: {local_model_dir}")

#device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
#print ("device is ", device)

device = "cpu"
encode_kwargs = {'normalize_embeddings': False}

embedding_model = HuggingFaceEmbeddings(
    model_name=local_model_dir,
    model_kwargs={'device': device},
    encode_kwargs=encode_kwargs
)



try:
    print("Testing embedding model with a simple query...")
    faulthandler.enable()
    embedding = embedding_model.embed_query("hello world")
    print(f"Embedding generated successfully: {embedding}")
    print(f"Embedding dimension: {len(embedding)}")
    vector_dim = len(embedding)
except Exception as e:
    print(f"Error during embedding generation: {e}")

index = faiss.IndexFlatL2(vector_dim)

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

uuids = [str(uuid4()) for _ in range(len(docs))]


vector_store.add_documents(documents=docs, ids=uuids)

query = "What is LLM?"
results= vector_store.similarity_search(
    query,  # the search query
    k=3  # returns top 3 most relevant chunks of text
)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")



hf_auth = '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'

#model_id = 'meta-llama/Llama-2-13b-chat-hf'
#model_id ="microsoft/phi-2"
model_id =microsoft/phi-1_5

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto',
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
llm = HuggingFacePipeline(pipeline=generate_text)


memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000,output_key="answer")

_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.

Chat History:
{chat_history}

Question:
{question}

Search query:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

sys_prompt = """You are a helpful AI Assistant.
Answer questions by considering following context.
{context}

Answer the following question.
{question}

Answer:
"""

prompt = PromptTemplate(
    template=sys_prompt, input_variables=["context", "question"]
)





# Initialize the QA chain


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt" : prompt},
    verbose=True,
    memory=memory
)


#Initialize FastAPI app

app = FastAPI()

# Add a root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI RAG system!"}




# Define input model for FastAPI
class Query(BaseModel):
    query: str

# API endpoint for querying the RAG solution
@app.post("/query/")
def query_rag_system(query: Query):
    result = qa_chain({"question": query.query})
    return {
        "answer": result["answer"],
        "sources": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result["source_documents"]
        ],
    }

# Create a public URL for your local server
#ngrok_tunnel = ngrok.connect(8000)
#print("Public URL:", ngrok_tunnel.public_url)


# Define the FastAPI endpoint URL (replace with your Ngrok URL)
#url = "https://62e6-34-126-85-132.ngrok-free.app"

# Define the JSON payload
'''
payload = {
    "query": "What is AI?"
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    print("Response:")
    print(response.json())
else:
    print(f"Failed to connect. Status code: {response.status_code}")
    print(response.text)
'''

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
