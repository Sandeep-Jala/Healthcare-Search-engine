
HEALTHCARE DATA SEARCH ENGINE
Information Retrieval Pipeline

Overview:
A modular end-to-end Healthcare Data Search Engine using Retrieval-Augmented Generation (RAG). This system preprocesses medical documents, builds vector and traditional indexes, and retrieves the most relevant information through hybrid ranking.

Repository Structure:
.
├── requirements.txt
├── main.py
├── main2.py
├── main_with_vector_ranker.py
├── Document_retriver.py
├── RAG.py
├── l2r.py
├── indexing.py
├── document_preprocessor.py
├── vector_ranker.py
├── relevance.py
├── ranker.py

File Descriptions:
- requirements.txt: Lists all Python dependencies.
- document_preprocessor.py: Cleans and normalizes raw healthcare text.
- indexing.py: Builds FAISS vector indexes and inverted indexes.
- vector_ranker.py: Implements bi-encoder embeddings and FAISS search.
- l2r.py: Defines and trains learning-to-rank models.
- relevance.py: Computes classical relevance scores (BM25, TF–IDF).
- ranker.py: Combines vector, L2R, and classical scores for hybrid ranking.
- Document_retriver.py: High-level class to fetch top-k documents for a query.
- RAG.py: Runs the full Retrieval-Augmented Generation loop with an LLM.
- main.py & main2.py: Prototype scripts (with/without bi-encoder).
- main_with_vector_ranker.py: Primary entry point for indexing and querying.

Quick Start:
1. Clone repository:
   git clone <repo_url>
   cd <repo_folder>

2. Install dependencies:
   pip install -r requirements.txt

3. Build the index:
   python main_with_vector_ranker.py --mode index --input_dir ./data/raw_docs/ --index_path ./data/faiss.index

4. Query the index:
   python main_with_vector_ranker.py --mode query --index_path ./data/faiss.index --query "What treatments exist for hypertension?" --top_k 5

5. End-to-end RAG:
   python RAG.py --index_path ./data/faiss.index --query "How is diabetes diagnosed?" --top_k 3 --llm_model gpt-3.5-turbo

Configuration:
--mode        : index or query
--input_dir   : Path to raw document folder
--index_path  : Path to save/load FAISS index
--query       : Search query string
--top_k       : Number of top results to return
--llm_model   : LLM model name for RAG (e.g., gpt-3.5-turbo)

Extending:
- Swap in different sentence-transformer embedder in vector_ranker.py
- Add new ranking features in relevance.py or l2r.py
- Customize prompt templates in RAG.py

