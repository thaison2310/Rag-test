# RAG Chatbot for Cosmetic Retail (Gemini)

What's included:
- `src/load_from_mongo.py`: load records from MongoDB collections
- `src/prepare_documents.py`: convert DB records to text documents with metadata
- `src/embeddings_gemini.py`: embedding client placeholder (instructions to wire Gemini)
- `src/build_index.py`: build a FAISS index from embeddings and save it
- `src/chatbot_cli.py`: simple CLI to run a RAG query and call Gemini for generation (placeholder)

Setup
1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

2. Create a `.env` file with MongoDB connection string and Gemini/API key (name these variables `MONGO_URI` and `GEMINI_API_KEY`).
3. Run the pipeline:

```powershell
python src\build_index.py
python src\chatbot_cli.py
```
