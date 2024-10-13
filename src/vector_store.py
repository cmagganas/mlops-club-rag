import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.readers.base import BaseReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

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
    
# Sourcery
# Hello! This is Sourcery's coding assistant. What can I do for you?

# Execute Recipe: Generate Docstrings

# selection: workflow.py:50-72
# """
# Retrieves relevant nodes from a database based on a provided query.

# This function serves as the entry point for the Retrieval-Augmented Generation (RAG) process, triggered by a StartEvent containing a query. It stores the query in the global context and retrieves nodes from the specified index if available.

# Args:
#     ctx (Context): The context in which the retrieval operation is performed.
#     ev (StartEvent): The event containing the query and index for the retrieval.

# Returns:
#     RetrieverEvent | None: An event containing the retrieved nodes, or None if the query is empty or the index is not provided.

# Raises:
#     None: This function does not raise any exceptions.

# Examples:
#     await retrieve(ctx, StartEvent(query="example query", index=my_index))
# """


# Execute Recipe: Generate Tests

# selection: vector_store.py:13-27
# import pytest
# import os
# from vector_store import __init__ as init_function

# @pytest.mark.parametrize(
#     "input_dir, recursive, exclude, file_metadata, pinecone_api_key, index_name, expected",
#     [
#         # Happy path test cases
#         ("out", True, ["*.srt"], lambda x: {"source": x}, "test_api_key", "test_index", 
#          {"input_dir": "out", "recursive": True, "exclude": ["*.srt"], "file_metadata": lambda x: {"source": x}, "pinecone_api_key": "test_api_key", "index_name": "test_index"}),
#         ("data", False, ["*.txt"], lambda x: {"type": "text"}, "another_key", "another_index", 
#          {"input_dir": "data", "recursive": False, "exclude": ["*.txt"], "file_metadata": lambda x: {"type": "text"}, "pinecone_api_key": "another_key", "index_name": "another_index"}),

#         # Edge case test cases
#         ("", False, [], lambda x: {}, "", None, 
#          {"input_dir": "", "recursive": False, "exclude": [], "file_metadata": lambda x: {}, "pinecone_api_key": "", "index_name": None}),
#         (None, None, None, None, None, None, 
#          {"input_dir": None, "recursive": None, "exclude": None, "file_metadata": None, "pinecone_api_key": None, "index_name": None}),

#         # Error case test cases
#         (123, True, ["*.srt"], lambda x: {"source": x}, "test_api_key", "test_index", 
#          TypeError),  # input_dir should be a string
#         ("out", "yes", ["*.srt"], lambda x: {"source": x}, "test_api_key", "test_index", 
#          TypeError),  # recursive should be a boolean
#         ("out", True, "*.srt", lambda x: {"source": x}, "test_api_key", "test_index", 
#          TypeError),  # exclude should be a list
#     ],
#     ids=[
#         "happy_path_default_values",
#         "happy_path_custom_values",
#         "edge_case_empty_values",
#         "edge_case_none_values",
#         "error_case_invalid_input_dir",
#         "error_case_invalid_recursive",
#         "error_case_invalid_exclude",
#     ]
# )
# def test_init_function(input_dir, recursive, exclude, file_metadata, pinecone_api_key, index_name, expected):
#     # Arrange
#     if isinstance(expected, dict):
#         # Act
#         result = init_function(input_dir, recursive, exclude, file_metadata, pinecone_api_key, index_name)

#         # Assert
#         assert result.input_dir == expected["input_dir"]
#         assert result.recursive == expected["recursive"]
#         assert result.exclude == expected["exclude"]
#         assert result.file_metadata == expected["file_metadata"]
#         assert result.pinecone_api_key == expected["pinecone_api_key"]
#         assert result.index_name == expected["index_name"]
#     else:
#         # Act & Assert
#         with pytest.raises(expected):
#             init_function(input_dir, recursive, exclude, file_metadata, pinecone_api_key, index_name)
# Explanation:
# Happy Path Tests: These tests check if the function behaves correctly with typical input values.
# Edge Cases: These tests handle unusual but valid inputs, such as empty strings or None.
# Error Cases: These tests ensure that the function raises appropriate exceptions for invalid inputs.
# Parametrization: Each test case is uniquely identified for clarity and debugging.
# Arrange-Act-Assert Pattern: The pattern is followed, with the Arrange section omitted when all inputs are provided directly.