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


from llama_index.core import VectorStoreIndex
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
        print("Starting retrieve step")
        query = ev.get("query")
        pc = Pinecone(api_key=os.environ["PINECONE_API"])

        pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        index = VectorStoreIndex.from_vector_store(vector_store)

        if not query:
            print("No query provided")
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
        print("Finished retrieve step")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        print("Starting rerank step")
        # Rerank the nodes
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm=OpenAI(model="gpt-4o-mini")
        )
        query = await ctx.get("query", default=None)
        print(f"Query for reranking: {query}", flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=query
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        print("Finished rerank step")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        print("Starting synthesize step")
        """Return a streaming response using reranked nodes."""
        llm = OpenAI(model="gpt-4o-mini")
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        print("Finished synthesize step")
        return StopEvent(result=response)


# async def main():
#     print("Initializing workflow")
#     w = RAGWorkflow()

#     # Run a query
#     print("Running query")
#     await init_tracing()
#     result = await w.run(query="How was Llama2 trained?")
#     async for chunk in result.async_response_gen():
#         print(chunk, end="", flush=True)
#     print("Query completed")

# if __name__ == "__main__":
#     print("Starting main")
#     asyncio.run(main())
#     print("Main completed")
    
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


async def main():
    print("Initializing workflow")
    w = RAGWorkflow()

    # Run a query
    print("Running query")
    # Add Phoenix
    span_phoenix_processor = SimpleSpanProcessor(
        HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
    )
    # Add them to the tracer
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

    # Instrument the application
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # return result
    result = await w.run(query="How was Llama2 trained?")
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)
    print("Query completed")

if __name__ == "__main__":
    print("Starting main")
    asyncio.run(main())
    print("Main completed")
