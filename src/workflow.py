import asyncio
import os
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from trace import init_tracing
from vector_store import VectorStore

from pinecone import Pinecone


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore


class RAGWorkflow(Workflow):
    # @step
    # async def load(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
    #     """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
    #     dirname = ev.get("dirname")
    #     if not dirname:
    #         return None

    #     # Use VectorStore to get the index
    #     vector_store = VectorStore(input_dir=dirname)
    #     index = vector_store.populate_index()
        
    #     return LoadIndexEvent(index=index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        pc = Pinecone(api_key=os.environ["PINECONE_API"])

        pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        index = VectorStoreIndex.from_vector_store(vector_store)

        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm=OpenAI(model="gpt-4o-mini")
        )
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = OpenAI(model="gpt-4o-mini")
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)


async def main():
    w = RAGWorkflow()

    # Run a query
    result = await w.run(query="How was Llama2 trained?")
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    init_tracing()
    asyncio.run(main())
