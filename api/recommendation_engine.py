import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# LOAD FAISS + METADATA + MODEL

print("Loading FAISS index...")
index = faiss.read_index("shl_faiss.index")

print("Loading metadata...")
with open("shl_metadata.pkl", "rb") as f:
    meta = pickle.load(f)

df = meta["df"]
model_name = meta["model_name"]
model = SentenceTransformer(model_name)

print("Loaded model:", model_name)

# Get top-k results from FAISS
def semantic_search(query, top_k=20):
    query_emb = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx].to_dict()
        results.append(row)

    return results



#  LLM Re-ranking 

import os
from google.ai.generativelanguage_v1beta import GenerativeServiceClient
from google.ai.generativelanguage_v1beta.types import GenerateContentRequest, Content
from google.api_core.client_options import ClientOptions

# Read from environment (Render + local)
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Add it in Render.")

client = GenerativeServiceClient(
    client_options=ClientOptions(api_key=API_KEY)
)

def rerank_with_llm(query, retrieved_items, top_k=10):
    try:
        context = ""
        for i, item in enumerate(retrieved_items):
            context += (
                f"[{i}] Name: {item['Assessment Name']}\n"
                f"Description: {item['Description']}\n"
                f"Test Type: {item['Test Type']}\n\n"
            )

        prompt = f"""
You are an SHL assessment ranking model. Rank assessments for the job query.

JOB QUERY:
{query}

ASSESSMENTS:
{context}

Return ONLY a comma-separated list of indexes.
"""

        request = GenerateContentRequest(
            model="models/gemini-2.5-flash",
            contents=[Content(parts=[{"text": prompt}])]
        )

        response = client.generate_content(request)

        ranked_indexes = response.candidates[0].content.parts[0].text.strip()
        ranked_indexes = [int(x) for x in ranked_indexes.split(",")]

        return [retrieved_items[i] for i in ranked_indexes[:top_k]]

    except Exception as e:
        print("Gemini reranking failed → Using raw top-k. Error:", e)
        return retrieved_items[:top_k]



# Format results 

def format_output(items):
    formatted = []

    for item in items:
        # Clean duration safely 
        duration_value = item.get("Duration", None)

        try:
            if duration_value is None or str(duration_value).lower() == "nan" or duration_value == "":
                duration_clean = None
            else:
                duration_clean = int(float(duration_value))
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


# MAIN PIPELINE FUNCTION TO CALL FROM API

def recommend(query, use_llm=True):
    retrieved = semantic_search(query, top_k=20)

    if use_llm:
        final_items = rerank_with_llm(query, retrieved, top_k=10)
    else:
        final_items = retrieved[:10]

    output = format_output(final_items)
    return output


