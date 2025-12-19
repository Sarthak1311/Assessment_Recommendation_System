# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn

# # Import recommendation pipeline
# from api.recommendation_engine import recommend

# app = FastAPI()

# # Health Check Endpoint

# @app.get("/health")
# def health():
#     return {"status": "healthy"}

# # Request Model

# class QueryRequest(BaseModel):
#     query: str


# # Recommendation Endpoint

# @app.post("/recommend")
# def get_recommendations(data: QueryRequest):
#     query = data.query
#     response = recommend(query, use_llm=True)   # set True if you want LLM reranker
#     return response


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"msg": "Render OK"}
