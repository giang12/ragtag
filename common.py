import os
from dotenv import load_dotenv
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import (
    StorageContext
)
import psycopg2
from sqlalchemy import make_url

load_dotenv()

def set_local_models(Settings, llm_model: str = "llama3-chatqa:8b", embeb_model: str = "nomic-embed-text:v1.5"):
    #default sentencesplitter
    Settings.node_parser = HierarchicalNodeParser.from_defaults()
    # default tiktoken
    # Settings.tokenizer
    # use Nomic
    Settings.embed_model = OllamaEmbedding(
        model_name=embeb_model,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )
    # setting a high request timeout in case you need to build an answer based on a large set of documents
    Settings.llm = Ollama(model=llm_model, request_timeout=120)

connection_string = "postgresql://postgres:postgres@localhost:5432"
db_name = "postgres"
conn = psycopg2.connect(connection_string)
conn.autocommit = True
db_url = make_url(connection_string)

def get_vector_store():
    vector_store = PGVectorStore.from_params(
        host=db_url.host,
        port=db_url.port,
        database=db_name,
        user=db_url.username,
        password=db_url.password,
        table_name="knowledge_base_vectors",
        # embed dim for this model can be found on https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        embed_dim=int(os.environ.get("EMBED_DIM", 768))
    )
    return vector_store

def get_doc_store():
    store = PostgresDocumentStore.from_params(
        host=db_url.host,
        port=db_url.port,
        database=db_name,
        user=db_url.username,
        password=db_url.password,
        table_name="knowledge_base_docs",
    )
    return store

def get_storage_context():
    storage_context = StorageContext.from_defaults(docstore=get_doc_store(), vector_store=get_vector_store())

    return storage_context