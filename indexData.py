import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    # abstraction that integrates various storage backends
    VectorStoreIndex
)
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

import os
from common import set_local_models, get_storage_context
# ! comment if you don't want to see everything that's happening under the hood
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def sanitize(documents):
    for document in documents:
        document.text = document.text.replace('\x00', '')

# time the execution
start = datetime.now()

load_dotenv()

set_local_models(Settings, os.environ.get("LLM_MODEL"), os.environ.get("EMBED_MODEL"))

documents = SimpleDirectoryReader(os.environ.get("KNOWLEDGE_BASE_DIR"), recursive=True).load_data()
sanitize(documents)

storage_context = get_storage_context()

nodes = Settings.node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

storage_context.docstore.add_documents(nodes)

# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, show_progress=True
# )
index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    show_progress=True
)
end = datetime.now()
# print the time it took to execute the script
print(f"Index time: {(end - start).total_seconds()}")
