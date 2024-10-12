from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from pinecone import Pinecone, ServerlessSpec, list_indexes
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

class VectorStore(BaseReader):
    def __init__(self, 
        input_dir="./out", 
        recursive=True, 
        exclude=["*.srt"], 
        file_metadata=lambda x: {"source": x},
        pinecone_api_key=os.environ['PINECONE_API'], 
        index_name=os.getenv("PINECONE_INDEX_NAME")):


        self.input_dir = input_dir
        self.recursive = recursive
        self.exclude = exclude
        self.file_metadata = file_metadata
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name

    def load_data(self):
        reader = SimpleDirectoryReader(
            input_dir=self.input_dir,
            recursive=self.recursive,
            exclude=self.exclude,
            file_metadata=self.file_metadata
        )
        return reader.load_data()

    def instantiate_index(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        if self.index_name not in pc.list_indexes().names():
            
            pc.create_index(
            name=self.index_name,
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
            return pc.Index(self.index_name)
        else:
            return pc.Index(self.index_name)
    
    def populate_index(self):
        documents = self.load_data()
        index = self.instantiate_index()

        vector_store = PineconeVectorStore(
            pinecone_index=index,
            add_sparse_vector=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

    def query_engine(self):
        index = self.populate_index()
        return index.as_query_engine()


if __name__ == "__main__":  
    vc = VectorStore()
    qe = vc.query_engine()

    print(qe.query("What is MLOps?"))