from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


def init_tracing():
    # Add Phoenix
    span_phoenix_processor = SimpleSpanProcessor(
        HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
    )
    # Add them to the tracer
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

    # Instrument the application
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
