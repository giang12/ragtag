# simple local rag with llama-index and ollama
docker compose up -d
download models via ollama
add datasets under ./data
python3 ./indexData
python3 -m streamlit run ./app