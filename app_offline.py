import streamlit as st
import numpy as np
import json, ollama
from sentence_transformers import SentenceTransformer

# Load local embeddings
with open("embeddings.json", "r") as f:
    data = json.load(f)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_embeddings(query_embedding, top_k=3):
    scores = []
    for item in data:
        sim = cosine_similarity(query_embedding, item["embedding"])
        scores.append((sim, item["text"], item.get("file", "unknown")))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# Local embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Offline Diocesan Policy Q&A")
st.write("Fully local search and conversation across all diocesan manuals.")

query = st.text_input("Your question:")

if query:
    q_emb = embedder.encode(query).tolist()
    results = search_embeddings(q_emb)
    context = "\n\n".join([r[1] for r in results])
    prompt = f"Use the following diocesan policy excerpts to answer clearly and accurately.\n\n{context}\n\nQuestion: {query}"

    with st.spinner("Thinking..."):
        reply = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        answer = reply["message"]["content"]

    st.markdown("### Response")
    st.write(answer)

    st.markdown("---")
    st.markdown("#### Source Passages:")
    for score, text, filename in results:
        st.markdown(f"**{filename}**  \n{text[:400]}...")

