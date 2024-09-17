import os
from dotenv import load_dotenv
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore

import psycopg2
from sqlalchemy import make_url

load_dotenv()

def set_local_models(Settings, llm_model: str = "llama3-chatqa:8b", embeb_model: str = "nomic-embed-text:v1.5"):
    # use Nomic
    Settings.embed_model = OllamaEmbedding(
        model_name=embeb_model,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )
    # setting a high request timeout in case you need to build an answer based on a large set of documents
    Settings.llm = Ollama(model=llm_model, request_timeout=120)


def get_vector_store():
    # of course, you can store db credentials in some secret place if you want
    connection_string = "postgresql://postgres:postgres@localhost:5432"
    db_name = "postgres"
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True
    db_url = make_url(connection_string)

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=db_url.host,
        password=db_url.password,
        port=db_url.port,
        user=db_url.username,
        table_name="knowledge_base_vectors",
        # embed dim for this model can be found on https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        embed_dim=int(os.environ.get("EMBED_DIM", 768))
    )
    return vector_store