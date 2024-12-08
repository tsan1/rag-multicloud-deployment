# Use an official Python image as the base
FROM python:3.12.1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set the environment variables (optional, can also be passed during runtime)
ENV DIRECTORY_PATH=example_data/
ENV EMBED_MODEL_ID=sentence-transformers/all-mpnet-base-v2
ENV HF_AUTH_TOKEN=""
ENV GENERATIVE_MODEL_ID=microsoft/phi-1_5
ENV DEVICE=cpu
ENV CHUNK_SIZE=1000
ENV CHUNK_OVERLAP=20
ENV PORT=8000

# Expose the application's port
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
# Command to run the main Python script
CMD ["python", "rag_main.py"]
