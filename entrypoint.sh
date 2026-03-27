#!/bin/bash
set -e

# Step 1: Build vector store
python Ai-engine/vectorstore.py

# Step 2: Run RAG pipeline
python Ai-engine/ragpipeline.py

# Step 3: Start the Streamlit app
streamlit run Ai-engine/llama.py --server.port 7860 --server.address 0.0.0.0
