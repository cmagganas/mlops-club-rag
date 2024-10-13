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
    def __init__(
        self,
        input_dir="out",
        recursive=True,
        exclude=["*.srt"],
        file_metadata=lambda x: {"source": x},
        pinecone_api_key=os.environ["PINECONE_API"],
        index_name=os.getenv("PINECONE_INDEX_NAME"),
    ):
        self.input_dir = input_dir
        self.recursive = recursive
        self.exclude = exclude
        self.file_metadata = file_metadata
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name

    def load_data(self):
        print(f"Loading data from directory: {self.input_dir}")
        reader = SimpleDirectoryReader(
            input_dir=self.input_dir,
            recursive=self.recursive,
            exclude=self.exclude,
            file_metadata=self.file_metadata,
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

    def populate_index(self):
        # Check if the index already exists
        pc = Pinecone(api_key=self.pinecone_api_key)
        if pc.has_index(self.index_name):
            print(f"Using existing Pinecone index: {self.index_name}")
            pinecone_index = pc.Index(self.index_name)
        else:
            # Create Index
            print(f"Creating new Pinecone index: {self.index_name}")
            documents = self.load_data()
            pinecone_index = self.instantiate_index()

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            self.load_data(), storage_context=storage_context
        )
        return index

    def query_engine(self):
        index = self.populate_index()
        # Assuming VectorStoreIndex is the correct type that has the as_query_engine method
        if isinstance(index, VectorStoreIndex):
            return index.as_query_engine(vector_store_query_mode="hybrid")
        else:
            raise TypeError("The index object is not of type VectorStoreIndex")


if __name__ == "__main__":
    vc = VectorStore()
    qe = vc.query_engine()
    print(qe.query("How do I install oh my zsh?"))