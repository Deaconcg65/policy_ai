import streamlit as st
from openai import OpenAI
import os, json, numpy as np, textwrap

# --- Load API key ---
import os
from dotenv import load_dotenv
load_dotenv()


embeddings_path = "embeddings.json"

def build_embeddings():
    st.write("⚙️ Building embeddings from all manuals... please wait.")
    txt_files = [f for f in os.listdir() if f.endswith(".txt")]
    if not txt_files:
        st.error("No .txt files found in this folder.")
        return

    embeddings, total_chunks = [], 0
    for file in txt_files:
        with open(file, "r", encoding="latin-1") as f:
            text = f.read()
        chunks = textwrap.wrap(text, 1000)
        total_chunks += len(chunks)
        st.write(f"📘 {file}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            ).data[0].embedding
            embeddings.append({"text": chunk, "embedding": emb, "file": file})
            if i % 20 == 0:
                st.write(f"Embedding chunk {i+1}/{len(chunks)} for {file}")

    with open(embeddings_path, "w") as f:
        json.dump(embeddings, f)
    st.success(f"✅ Embedded {len(txt_files)} files, {total_chunks} chunks total.")

if not os.path.exists(embeddings_path):
    build_embeddings()

with open(embeddings_path, "r") as f:
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

st.title("Diocesan Policy Q&A")
st.write("Ask a question based on all diocesan policy manuals in this folder.")

query = st.text_input("Your question:")

if query:
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    results = search_embeddings(q_emb)
    context = "\n\n".join([r[1] for r in results])
    prompt = f"Use only the diocesan policy content below to answer the question. Cite the file name when relevant.\n\nContext:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a diocesan policy assistant. Quote and cite text from the given context only."},
            {"role": "user", "content": prompt}
        ]
    )

    st.markdown("### Response")
    st.write(response.choices[0].message.content)

    st.markdown("---")
    st.markdown("#### Source Passages:")
    for score, text, filename in results:
        st.markdown(f"**{filename}**  \n{text[:400]}...")

