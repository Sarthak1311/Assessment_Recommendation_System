# ===============================
# recommendation_engine.py
# ===============================

import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer


# CPU SAFETY (IMPORTANT FOR RENDER)

torch.set_num_threads(1)


# GLOBAL CACHED OBJECTS (LAZY LOADED)

_index = None
_model = None
_df = None
_gemini_model = None


# PATHS (ABSOLUTE, CLOUD-SAFE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH = os.path.join(BASE_DIR, "shl_faiss.index")
META_PATH = os.path.join(BASE_DIR, "shl_metadata.pkl")


# SAFETY CHECKS (FAIL FAST, CLEAR ERROR)

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Metadata file not found at {META_PATH}")


# LAZY RESOURCE LOADER

def load_resources():
    global _index, _model, _df

    if _index is None or _model is None or _df is None:
        print("🚀 Loading FAISS index & embedding model (lazy)...")

        # Load FAISS index
        _index = faiss.read_index(FAISS_PATH)

        # Load metadata
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)

        _df = meta["df"]
        model_name = meta["model_name"]

        # Load embedding model (HEAVY)
        _model = SentenceTransformer(model_name)

        print(f"Loaded embedding model: {model_name}")


# SEMANTIC SEARCH

def semantic_search(query, top_k=20):
    load_resources()   # ensures non-blocking startup

    query_emb = _model.encode(
        [query],
        normalize_embeddings=True
    )

    _, indices = _index.search(query_emb, top_k)

    results = []
    for idx in indices[0]:
        results.append(_df.iloc[idx].to_dict())

    return results


# GEMINI LLM (LAZY LOADED)

import google.genai as genai

def get_gemini_model():
    global _gemini_model

    if _gemini_model is None:
        API_KEY = os.getenv("GOOGLE_API_KEY")
        if not API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        print(" Gemini model loaded")

    return _gemini_model


# LLM RE-RANKING

def rerank_with_llm(query, retrieved_items, top_k=10):
    try:
        model = get_gemini_model()

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

        response = model.generate_content(prompt)

        ranked_indexes = [
            int(x.strip())
            for x in response.text.strip().split(",")
            if x.strip().isdigit()
        ]

        return [retrieved_items[i] for i in ranked_indexes[:top_k]]

    except Exception as e:
        print(" Gemini reranking failed → Using raw results:", e)
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


# MAIN ENTRY POINT (CALLED BY FASTAPI)

def recommend(query, use_llm=True):
    retrieved = semantic_search(query, top_k=20)

    if use_llm:
        final_items = rerank_with_llm(query, retrieved, top_k=10)
    else:
        final_items = retrieved[:10]

    return format_output(final_items)
