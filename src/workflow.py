#!/usr/bin/env python
# coding: utf-8

import nest_asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.utils.workflow import draw_all_possible_flows
from IPython.display import Markdown, display

nest_asyncio.apply()
load_dotenv()

PINECONE_API = os.environ["PINECONE_API"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
OTEL_EXPORTER_OTLP_HEADERS = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
PHOENIX_CLIENT_HEADERS = os.environ["PHOENIX_CLIENT_HEADERS"]
PHOENIX_COLLECTOR_ENDPOINT = os.environ["PHOENIX_COLLECTOR_ENDPOINT"]

pc = Pinecone(api_key=os.environ["PINECONE_API"])
index_name = os.environ["PINECONE_INDEX_NAME"]
pinecone_index = pc.Index(index_name)

DEFAULT_RAG_PROMPT = PromptTemplate(
    template="""Use the provided context to answer the question. If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
)

class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""
    pass

class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""
    retrieved_nodes: list[NodeWithScore]

class AugmentGenerateEvent(Event):
    """Query event. Queries given relevant text and search text."""
    relevant_text: str
    search_text: str

class WorkflowRAG(Workflow):
    @step
    async def initialize_index(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Initializing index."""
        pc = Pinecone(api_key=os.environ["PINECONE_API"])
        index_name = os.environ["PINECONE_INDEX_NAME"]
        pinecone_index = pc.Index(index_name)

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context
        )
        return StopEvent(result=index)

    @step
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> PrepEvent | None:
        """Prepare for retrieval."""
        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        index = ev.get("index")

        llm = OpenAI(model="gpt-4o-mini")
        await ctx.set("rag_pipeline", QueryPipeline(
            chain=[DEFAULT_RAG_PROMPT, llm]
        ))

        await ctx.set("llm", llm)
        await ctx.set("index", index)

        await ctx.set("query_str", query_str)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        return PrepEvent()

    @step
    async def retrieve(
        self, ctx: Context, ev: PrepEvent
    ) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")

        if query_str is None:
            return None

        index = await ctx.get("index", default=None)
        if not (index):
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        retriever: BaseRetriever = index.as_retriever(
            **retriever_kwargs
        )
        result = retriever.retrieve(query_str)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)

    @step
    async def augment_and_generate(self, ctx: Context, ev: RetrieveEvent) -> StopEvent:
        """Get result with relevant text."""
        relevant_nodes = ev.retrieved_nodes
        relevant_text = "\n".join([node.get_content() for node in relevant_nodes])
        query_str = await ctx.get("query_str")

        relevancy_pipeline = await ctx.get("rag_pipeline")

        relevancy = relevancy_pipeline.run(
                context=relevant_text, question=query_str
        )

        return StopEvent(result=relevancy.message.content)

async def main(question: str):
    draw_all_possible_flows(
        WorkflowRAG, filename="wf_rag_workflow.html"
    )

    rag_workflow = WorkflowRAG()
    index = await rag_workflow.run()

    response = await rag_workflow.run(
        query_str=question,
        index=index,
    )
    markdown_content = str(response)
    
    return markdown_content

if __name__ == "__main__":
    import argparse
    import asyncio
    from rich.console import Console
    from rich.markdown import Markdown
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the RAG workflow with a question."
    )
    parser.add_argument(
        "-question",
        type=str,
        default="How do I install oh my zsh?",
        help="The question to ask the RAG workflow.",
    )
    args = parser.parse_args()
    
    # Add Phoenix
    span_phoenix_processor = SimpleSpanProcessor(
        HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
    )
    # Add them to the tracer
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

    # Instrument the application
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    output = asyncio.run(main(args.question))
    
    # Output the markdown content to the terminal using rich
    console = Console()
    md = Markdown(output)
    console.print(md)
