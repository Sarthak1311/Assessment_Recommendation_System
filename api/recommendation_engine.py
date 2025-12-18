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

def rerank_with_llm(query, retrieved_items, top_k=10):
    """
    You can use OpenAI, Gemini, or any LLM.
    I'll provide a clean OpenAI example below.
    """

    try:
        from openai import OpenAI
        client = OpenAI()

        context = "\n\n".join(
            [f"[{i}] {item['name']} — {item['description']}" for i, item in enumerate(retrieved_items)]
        )

        prompt = f"""
        You are a ranking model for SHL assessment recommendations.

        Query:
        {query}

        Below is a list of candidate assessments. Rank them from most relevant to least relevant.

        {context}

        Return ONLY a list of ranked indexes (like: 3,0,1,2,...)
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        ranked_order = response.choices[0].message.content.strip()
        ranked_order = [int(x) for x in ranked_order.split(",")]

        reranked = [retrieved_items[i] for i in ranked_order[:top_k]]
        return reranked

    except Exception as e:
        print("LLM re-ranking failed, using raw results:", e)
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


