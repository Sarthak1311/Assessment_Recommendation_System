import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


# GLOBAL CACHED OBJECTS (lazy)

_index = None
_model = None
_df = None


# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "shl_faiss.index")
META_PATH = os.path.join(BASE_DIR, "shl_metadata.pkl")


# LAZY LOADER

def load_resources():
    global _index, _model, _df

    if _index is None or _model is None or _df is None:
        print(" Loading FAISS index & embedding model (lazy)...")

        # Load FAISS
        _index = faiss.read_index(FAISS_PATH)

        # Load metadata
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)

        _df = meta["df"]
        model_name = meta["model_name"]

        # Load embedding model (heavy)
        _model = SentenceTransformer(model_name)

        print(f" Loaded model: {model_name}")


# SEMANTIC SEARCH

def semantic_search(query, top_k=20):
    load_resources()   # <-- KEY FIX

    query_emb = _model.encode([query], normalize_embeddings=True)
    _, indices = _index.search(query_emb, top_k)

    results = []
    for idx in indices[0]:
        results.append(_df.iloc[idx].to_dict())

    return results


# GEMINI LLM RE-RANKING

import google.generativeai as genai
import os

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment variables")

genai.configure(api_key=API_KEY)

gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")


def rerank_with_llm(query, retrieved_items, top_k=10):
    try:
        context = ""
        for i, item in enumerate(retrieved_items):
            context += (
                f"[{i}] Name: {item.get('Assessment Name', '')}\n"
                f"Description: {item.get('Description', '')}\n"
                f"Test Type: {item.get('Test Type', '')}\n\n"
            )

        prompt = f"""
You are an SHL assessment ranking model.

JOB QUERY:
{query}

ASSESSMENTS:
{context}

Return ONLY a comma-separated list of indexes.
"""

        response = gemini_model.generate_content(prompt)
        ranked_indexes = response.text.strip()
        ranked_indexes = [int(x) for x in ranked_indexes.split(",")]

        return [retrieved_items[i] for i in ranked_indexes[:top_k]]

    except Exception as e:
        print("Gemini reranking failed → Using raw results. Error:", e)
        return retrieved_items[:top_k]



# OUTPUT FORMATTER

def format_output(items):
    formatted = []

    for item in items:
        duration_value = item.get("Duration", None)

        try:
            duration_clean = int(float(duration_value)) if duration_value else None
        except:
            duration_clean = None

        formatted.append({
            "url": item.get("Link", ""),
            "name": item.get("Assessment Name", ""),
            "description": item.get("Description", ""),
            "duration": duration_clean,
            "adaptive_support": item.get("Adaptive/IRT (Yes/No)", "Unknown"),
            "remote_support": item.get("Remote Support (Yes/No)", "Unknown"),
            "test_type": [item.get("Test Type", "Unknown")]
        })

    return {"recommended_assessments": formatted}

# MAIN ENTRY POINT

def recommend(query, use_llm=True):
    retrieved = semantic_search(query, top_k=20)

    if use_llm:
        final_items = rerank_with_llm(query, retrieved, top_k=10)
    else:
        final_items = retrieved[:10]

    return format_output(final_items)
