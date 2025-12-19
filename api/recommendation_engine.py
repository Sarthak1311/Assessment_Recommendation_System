import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
import google.genai as genai

# -------------------------------------------------
# CPU SAFETY (RENDER)
# -------------------------------------------------
torch.set_num_threads(1)

# -------------------------------------------------
# GLOBAL CACHED OBJECTS (LAZY)
# -------------------------------------------------
_index = None
_model = None
_df = None
_gemini_model = None

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "shl_faiss.index")
META_PATH = os.path.join(BASE_DIR, "shl_metadata.pkl")

# -------------------------------------------------
# LAZY RESOURCE LOADER (SAFE)
# -------------------------------------------------
def load_resources():
    global _index, _model, _df

    if _index is not None and _model is not None and _df is not None:
        return

    print("🚀 Lazy loading FAISS + model...")

    # 🔑 checks moved INSIDE function
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found at {META_PATH}")

    _index = faiss.read_index(FAISS_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    _df = meta["df"]
    model_name = meta["model_name"]

    _model = SentenceTransformer(model_name)

    print(f"✅ Loaded embedding model: {model_name}")

# -------------------------------------------------
# SEMANTIC SEARCH
# -------------------------------------------------
def semantic_search(query, top_k=20):
    load_resources()

    query_emb = _model.encode(
        [query],
        normalize_embeddings=True
    )

    _, indices = _index.search(query_emb, top_k)

    return [_df.iloc[i].to_dict() for i in indices[0]]

# -------------------------------------------------
# GEMINI (LAZY)
# -------------------------------------------------
def get_gemini_model():
    global _gemini_model

    if _gemini_model is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        print("✅ Gemini model loaded")

    return _gemini_model

# -------------------------------------------------
# LLM RERANK
# -------------------------------------------------
def rerank_with_llm(query, retrieved_items, top_k=10):
    try:
        model = get_gemini_model()

        context = ""
        for i, item in enumerate(retrieved_items):
            context += (
                f"[{i}] Name: {item.get('Assessment Name','')}\n"
                f"Description: {item.get('Description','')}\n"
                f"Test Type: {item.get('Test Type','')}\n\n"
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

        indexes = [
            int(x.strip())
            for x in response.text.split(",")
            if x.strip().isdigit()
        ]

        return [retrieved_items[i] for i in indexes[:top_k]]

    except Exception as e:
        print("⚠️ Gemini failed, using raw results:", e)
        return retrieved_items[:top_k]

# -------------------------------------------------
# FORMAT OUTPUT
# -------------------------------------------------
def format_output(items):
    formatted = []

    for item in items:
        try:
            duration = int(float(item.get("Duration", "")))
        except:
            duration = None

        formatted.append({
            "url": item.get("Link", ""),
            "name": item.get("Assessment Name", ""),
            "description": item.get("Description", ""),
            "duration": duration,
            "adaptive_support": item.get("Adaptive/IRT (Yes/No)", "Unknown"),
            "remote_support": item.get("Remote Support (Yes/No)", "Unknown"),
            "test_type": [item.get("Test Type", "Unknown")]
        })

    return {"recommended_assessments": formatted}

# -------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------
def recommend(query, use_llm=True):
    retrieved = semantic_search(query)

    if use_llm:
        retrieved = rerank_with_llm(query, retrieved)

    return format_output(retrieved)
