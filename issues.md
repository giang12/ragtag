https://github.com/run-llama/llama_index/issues/10751 NULL character

Can't load model with SentenceTransformers 3.0.1 AttributeError: 'LatentAttentionConfig' object has no attribute '_attn_implementation_internal'
downgrade -> transformers==4.43.4 https://huggingface.co/nvidia/NV-Embed-v1/discussions/50

only store leafNodes in index store if using HierarchicalNodeParser https://github.com/run-llama/llama_index/issues/12603 
https://github.com/run-llama/llama_docs_bot/blob/main/5_retrieval/5_retrieval.ipynb