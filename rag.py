from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    # function to create better responses
    get_response_synthesizer,
    Settings,
    VectorStoreIndex
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import AutoMergingRetriever

import os

from common import (
    set_local_models, 
    get_vector_store,
    get_storage_context
)

#https://huggingface.co/nvidia/Llama3-ChatQA-1.5-70B#when-context-is-available
TEXT_QA_PROMT_TMP = """
System: You are a knowledgeable assistant specialized in question-answering tasks,
Your goal is to provide accurate, consistent, and contextually relevant answers based on the given information and prior knowledge.
Please answer in the tone of Dark vador
Instructions:

Comprehension and Accuracy: Carefully read and comprehend the provided context to ensure accuracy in your response.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.
Consistentcy: Ensure to give the same answer to the same question. Only update answer when there are new information or context.
Remember if no context is provided please clearly state that "I'm guessing"

Context information is below.
"---------------------\n"
"{context_str}\n"
"---------------------\n"

User: {query_str}

Assistant:"""

TEXT_QA_PROMPT = PromptTemplate(
    TEXT_QA_PROMT_TMP, prompt_type=PromptType.QUESTION_ANSWER
)

def get_streamed_rag_query_engine():
    # time the execution
    start = datetime.now()

    load_dotenv()

    set_local_models(Settings, os.environ.get("LLM_MODEL"), os.environ.get("EMBED_MODEL"))

    # configure retriever
    retriever = AutoMergingRetriever(
        vector_retriever=VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=get_vector_store()),
            similarity_top_k=12,
        ), 
        storage_context=get_storage_context()
    )
    # retriever = VectorIndexRetriever(
    #         index=VectorStoreIndex.from_vector_store(vector_store=get_vector_store()),
    #         similarity_top_k=12,
    # ) 
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(streaming=True)
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # discarding nodes which similarity is below a certain threshold
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=.42)],
    )
    #custom prompt
    query_engine.update_prompts({"response_synthesizer:text_qa_template":TEXT_QA_PROMPT})

    end = datetime.now()
    # print the time it took to execute the script
    print(f"RAG time: {(end - start).total_seconds()}")

    return query_engine