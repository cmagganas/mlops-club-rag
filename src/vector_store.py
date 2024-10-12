from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from pinecone import Pinecone, ServerlessSpec, list_indexes
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os

class VectorStore(BaseReader):
    def __init__(self, input_dir, recursive=True, exclude=["*.srt"], file_metadata=lambda x: {"source": x}):
        self.input_dir = input_dir
        self.recursive = recursive
        self.exclude = exclude
        self.file_metadata = file_metadata
        self.api_key = os.environ["PINECONE_API_KEY"]
        self.index_name = os.environ["PINECONE_INDEX_NAME"]

    def load_data(self):
        reader = SimpleDirectoryReader(
            input_dir=self.input_dir,
            recursive=self.recursive,
            exclude=self.exclude,
            file_metadata=self.file_metadata
        )
        return reader.load_data()

    def instantiate_index(self):
        pc = Pinecone(api_key=self.api_key)
        if self.index_name not in list_indexes():
            
            pc.create_index(
            name=self.index_name,
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
            return pc.Index(self.index_name)
        else:
            return pc.Index(self.index_name)
    
    def upload_documents(self):
        documents = self.load_data()
        index = self.instantiate_index()
        index.upsert(documents)


documents = VectorStore(input_dir="./out").load_data()
print(documents[:5])