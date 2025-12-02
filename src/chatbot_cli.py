"""Simple CLI RAG chatbot: load FAISS index, retrieve, then call Gemini to answer."""
import os
import pickle
from dotenv import load_dotenv
from typing import List

import numpy as np
import faiss

from embeddings_gemini import GeminiClient

load_dotenv()

BASE = os.path.join(os.path.dirname(__file__), "..")
INDEX_PATH = os.path.join(BASE, "faiss_index.pkl")
DOCS_PATH = os.path.join(BASE, "documents.pkl")

# Configure for your domain
PRODUCT_DOMAIN = os.getenv("PRODUCT_DOMAIN")


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("Index or documents not found. Run src/build_index.py first.")
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs


def retrieve(query: str, embed_client: GeminiClient, index, docs, top_k: int = 5):
    q_emb = np.array(embed_client.get_embedding(query), dtype=np.float32).reshape(1, -1)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docs):
            continue
        results.append(docs[idx])
    return results


def build_context(retrieved: List[dict]) -> str:
    parts = []
    for r in retrieved:
        meta = r.get("metadata", {})
        collection = meta.get("collection")
        text = r.get("text")
        
        # Build source string with product link if available
        if collection == "products":
            product_id = meta.get("id")
            if product_id:
                product_link = f"{PRODUCT_DOMAIN}/all-products/product/{product_id}"
                parts.append(f"Sản phẩm (Link: {product_link}):\n{text}")
            else:
                parts.append(f"Sản phẩm:\n{text}")

        elif collection == "vouchers":
            parts.append(f"Khuyến mãi:\n{text}")
        else:
            parts.append(f"Nguồn ({collection}):\n{text}")
    
    return "\n\n".join(parts)


def main():
    client = GeminiClient()
    index, docs = load_index()

    print("=" * 60)
    print("🛍️  BEAUTÉ Chatbot - Hỏi đáp về sản phẩm, bình luận và khuyến mãi")
    print("=" * 60)
    print("Gõ câu hỏi của bạn (tiếng Việt). Gõ 'exit' hoặc 'quit' để thoát.\n")
    
    while True:
        q = input("Bạn: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Cảm ơn bạn đã sử dụng BEAUTÉ Chatbot!")
            break
        
        try:
            retrieved = retrieve(q, client, index, docs, top_k=5)
            context = build_context(retrieved)
            
            prompt = f"""Bạn là trợ lý thân thiện cho website bán mỹ phẩm BEAUTÉ. 
Trả lời câu hỏi của khách hàng dựa trên thông tin được cung cấp dưới đây.
Hãy trả lời ngắn gọn, hữu ích và bằng tiếng Việt.
Nếu có thông tin về sản phẩm, hãy bao gồm link sản phẩm.

Thông tin liên quan:
{context}

Câu hỏi: {q}

Trả lời:"""
            
            answer = client.generate(prompt=prompt, context="")
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"Lỗi: {e}\n")


if __name__ == "__main__":
    main()
