# Creating a Pinecone Index
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
load_dotenv()

PINECONE_API = os.environ["PINECONE_API"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

pc = Pinecone(api_key=PINECONE_API)

# Create a serverless index
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
pinecone_index = pc.Index(PINECONE_INDEX_NAME)


# Load documents, build the PineconeVectorStore

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore

# load documents
documents = SimpleDirectoryReader(
    input_dir="out",
    recursive=True,
    exclude=["*.srt"],
    file_metadata=lambda x: {"source": x}
)
# set add_sparse_vector=True to compute sparse vectors during upsert
from llama_index.core import StorageContext

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Environment variable OPENAI_API_KEY is not set")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents.load_data(), storage_context=storage_context
)

# Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(vector_store_query_mode="hybrid")
response = query_engine.query("How do I install oh my zsh?")