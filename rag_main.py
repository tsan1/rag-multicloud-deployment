from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from uuid import uuid4
import faulthandler
import transformers
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import faiss
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# **Step 1: Configuration and Initialization**
class Config:
    DIRECTORY_PATH: str = os.getenv("DIRECTORY_PATH", "example_data/")
    EMBED_MODEL_ID: str = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-mpnet-base-v2")
    HF_AUTH_TOKEN: str = os.getenv("HF_AUTH_TOKEN", "")
    GENERATIVE_MODEL_ID: str = os.getenv("GENERATIVE_MODEL_ID", "microsoft/phi-1_5")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 20))
    PORT: int = int(os.getenv("PORT", 8000))


# **Step 2: Load and Process Documents**
def load_documents(directory_path: str) -> List:
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No documents loaded. Check the directory path or file format.")
    
    for i, doc in enumerate(documents):
        if not doc.page_content.strip():
            print(f"Warning: Document {i} is empty.")
    
    return documents


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


# **Step 3: Initialize Embedding Model**
def initialize_embedding_model(embed_model_id: str, device: str) -> HuggingFaceEmbeddings:
    model = SentenceTransformer(embed_model_id)
    local_model_dir = "./embedding_model"
    model.save(local_model_dir)

    return HuggingFaceEmbeddings(
        model_name=local_model_dir,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': False}
    )


# **Step 4: Initialize Vector Store**
def initialize_vector_store(embedding_model, documents: List):
    faulthandler.enable()

    # Test embedding model
    embedding = embedding_model.embed_query("hello world")
    vector_dim = len(embedding)
    print ("vector dim is:", vector_dim)

    index = faiss.IndexFlatL2(vector_dim)
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    print ("Succesfully added documents to teh vector store")

    return vector_store


# **Step 5: Initialize Generative Model**
def initialize_llm(model_id: str, hf_auth_token: str):
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
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=True,
        temperature=0.0,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=generate_text)


# **Step 6: Initialize QA Chain**
def initialize_qa_chain(llm, retriever, memory):
    _template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.

    \n\n Chat History:
    {chat_history}

    \n\n Question:
    {question}

    Search query:
    """
    condense_question_prompt = PromptTemplate.from_template(_template)

    sys_prompt = """\n\n You are a helpful AI Assistant.
    Answer questions by considering the following context.
    {context}

    Answer the following question.
    {question}

    \n\n Answer:
    """

    prompt = PromptTemplate(
        template=sys_prompt,
        input_variables=["context", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False,
        memory=memory
    )


# **Step 7: Initialize FastAPI App**
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI RAG system!"}

class Query(BaseModel):
    query: str

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


# **Step 8: Main Entry Point**
if __name__ == "__main__":
    config = Config()
    print ("onfig.DIRECTORY_PATH", config.DIRECTORY_PATH)

    documents = load_documents(config.DIRECTORY_PATH)
    docs = split_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    embedding_model = initialize_embedding_model(config.EMBED_MODEL_ID, config.DEVICE)
    vector_store = initialize_vector_store(embedding_model, docs)
    retriever = vector_store.as_retriever()

    llm = initialize_llm(config.GENERATIVE_MODEL_ID, config.HF_AUTH_TOKEN)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        max_token_limit=1000,
        output_key="answer"
    )

    qa_chain = initialize_qa_chain(llm, retriever, memory)
    print ("until here")

    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
