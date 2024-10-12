# mlops-club-rag

RAG-A-THON: Taking Python to Production + Cloud Engineering

## Project Description

This project implements a Agentic Retrieval-Augmented Generation (RAG) system using LlamaIndex and Pinecone for vector storage. It's designed to process and query documents, with a focus on MLOps and cloud engineering concepts from Eric Riddoch's courses.

## Features

- Document ingestion and cleaning
- Vector storage using Pinecone
- Agentic RAG workflow implementation
- OpenTelemetry integration for tracing
- Query engine for asking questions about the ingested documents

## Installation

This project requires Python 3.12 or higher. To install the required dependencies, run:

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install uv
uv sync
```


## Usage

1. Data Cleaning:
   Run the data cleaning script to process SRT files:

   ```bash
   python src/data_cleaning.py
   ```

2. Vector Store:
   To populate the vector store and query it:

   ```bash
   python src/vector_store.py
   ```

3. RAG Workflow:
   To run the RAG workflow:
   
   ```bash
   python src/workflow.py
   ```

## Project Structure

- `src/data_cleaning.py`: Script for cleaning and processing SRT files
- `src/vector_store.py`: Handles vector storage using Pinecone
- `src/workflow.py`: Implements the RAG workflow
- `src/trace.py`: Sets up OpenTelemetry tracing