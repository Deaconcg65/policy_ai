import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os, json, numpy as np

# --- Load your API key from .env ---
load_dotenv()
client = OpenAI()

# --- Streamlit UI ---
st.title("Diocesan Policy Q&A")
st.write("Ask a question across diocesan manuals (Priest, Deacon, Lay Employee, Volunteer, etc.).")

# Optional role filter
role = st.selectbox(
    "Select role (optional):",
    ["All", "Priest", "Deacon", "Lay Employee", "Volunteer"]
)

# --- Load local embeddings ---
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

# --- Input box for questions ---
query = st.text_input("Your question:")

if query:
    # Create embedding for the user's query
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Retrieve top-matching passages
    results = search_embeddings(q_emb)

        # --- Optional role-based filtering ---
    if role != "All":
        role_keywords = {
            "Priest": ["priest", "clergy"],
            "Deacon": ["deacon", "diaconate"],
            "Lay Employee": ["employee", "handbook", "lay"],
            "Volunteer": ["volunteer", "safe environment", "virtus"],
        }
        keywords = role_keywords.get(role, [])
        results = [r for r in results if any(k in r[2].lower() for k in keywords)]

        if not results:
            st.warning(f"No passages found specifically for {role}. Showing best available context instead.")
            results = search_embeddings(q_emb)  # fallback to all manuals


    # Build combined context from filtered results
    context = "\n\n".join([r[1] for r in results])

    # --- Build the GPT prompt ---
    prompt = (
        "Use only the diocesan policy content below to answer the question. "
        "If the question could apply to multiple roles (priests, deacons, lay employees, volunteers, etc.), "
        "consider all relevant manuals and specify distinctions clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    # Generate the response using GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a diocesan policy assistant. "
                    "Answer clearly, quoting and citing only from the given context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    # --- Display results ---
    st.markdown("### Response")
    st.write(response.choices[0].message.content)

    st.markdown("---")
    st.markdown("#### Source Passages:")
    for score, text, filename in results:
        st.markdown(f"**{filename}**  \n{text[:400]}...")
