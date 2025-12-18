import streamlit as st
import requests

# ---------------------------
# CONFIG
# ---------------------------
API_URL = "http://127.0.0.1:8000/recommend"   # Change to deployed API later

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🔍 SHL Assessment Recommendation System")

st.write("""
Enter a job description or natural language query below.  
The system will recommend the most relevant SHL assessments.
""")

# ---------------------------
# USER INPUT BOX
# ---------------------------
query = st.text_area("Enter job description or query:", height=150)

if st.button("Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a query first.")
        st.stop()

    with st.spinner("Generating recommendations..."):
        # ---------------------------
        # SEND REQUEST TO FASTAPI
        # ---------------------------
        response = requests.post(API_URL, json={"query": query})

        if response.status_code != 200:
            st.error("API Error: " + str(response.text))
            st.stop()

        data = response.json()
        results = data.get("recommended_assessments", [])

        if len(results) == 0:
            st.warning("No results found.")
            st.stop()

        # ---------------------------
        # DISPLAY RESULTS IN TABLE
        # ---------------------------
        st.success(f"Found {len(results)} recommendations:")

        # create clean rows for display
        formatted_rows = []
        for item in results:
            formatted_rows.append({
                "Name": item["name"],
                "Description": item["description"],
                "Test Type": ", ".join(item["test_type"]),
                "Duration": item["duration"],
                "Remote Support": item["remote_support"],
                "Adaptive Support": item["adaptive_support"],
                "URL": item["url"]
            })

        st.table(formatted_rows)
