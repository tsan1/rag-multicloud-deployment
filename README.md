# Retrieval-Augmented Generation (RAG) System

This repository implements a Retrieval-Augmented Generation (RAG) system using FastAPI, LangChain, and HuggingFace. The system retrieves relevant documents and generates AI-powered answers based on the context and user queries.

## Features
- **Document Loading**: Load and index documents from a specified directory.
- **Embedding Creation**: Use pre-trained embeddings for efficient document retrieval.
- **Generative AI**: Answer user queries using state-of-the-art language models.
- **API Interface**: Expose the RAG system via RESTful API endpoints using FastAPI.
- **Modular Design**: Clean and modular codebase suitable for production.

---

## Requirements
- Python 3.9+
- FastAPI
- LangChain
- HuggingFace Transformers
- Sentence Transformers
- FAISS for vector indexing

---

## Setup

### 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Set Up the Environment
Create a .env file in the root directory and add the following variables:

```plaintext

DIRECTORY_PATH=example_data/
EMBED_MODEL_ID=sentence-transformers/all-mpnet-base-v2
HF_AUTH_TOKEN=
GENERATIVE_MODEL_ID=microsoft/phi-1_5
DEVICE=cpu
CHUNK_SIZE=1000
CHUNK_OVERLAP=20
PORT=8000
```


### 4. Add Documents
Place your PDF documents in the example_data/ directory (or the directory specified in DIRECTORY_PATH).

### 5. Run the Application
Start the FastAPI server:

```bash
python app.py 

Access the server at:
``` 
http://127.0.0.1:8000 


### Usage

Query the RAG System
Send a POST request to the /query/ endpoint with a JSON payload containing your query.

Example curl command:

``` bash
curl -X POST "http://127.0.0.1:8000/query/" \
-H "Content-Type: application/json" \
-d '{"query": "Explain the paper briefly."}'
```

API Endpoints

/
Method: GET
Description: Returns a welcome message.

/query/
Method: POST
Description: Queries the RAG system for an answer.
Input:

```json
 {"query": "Your question here"}  
```

Output:

``` json
{
  "answer": "Generated answer",
  "sources": [
    {
      "page_content": "Text snippet",
      "metadata": {"source": "metadata"}
    }
  ]
}
```


### Project Structure

```
<repo-name>/
│
├── app.py                # Main FastAPI application
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── example_data/         # Directory for PDF documents
├── README.md             # Project documentation
└── 
```

#### Contributing

Contributions are welcome! Follow these steps to contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a detailed description.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

