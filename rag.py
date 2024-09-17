from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    # function to create better responses
    get_response_synthesizer,
    Settings,
    VectorStoreIndex
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
import os

from common import set_local_models, get_vector_store

def get_streamed_rag_query_engine():
    # time the execution
    start = datetime.now()

    load_dotenv()

    set_local_models(Settings, os.environ.get("LLM_MODEL"), os.environ.get("EMBED_MODEL"))

    vector_store = get_vector_store()

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(streaming=True)
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # discarding nodes which similarity is below a certain threshold
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=.42)],
    )

    end = datetime.now()
    # print the time it took to execute the script
    print(f"RAG time: {(end - start).total_seconds()}")

    return query_engine