from openai import OpenAI
import json, os, textwrap

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Input text file
input_file = "Priest_Personnel_Manual_20221216.txt"

# Output JSON file
output_file = "embeddings.json"

# --- 1. Read text ---
with open(input_file, "r", encoding="latin-1") as f:
    text = f.read()

# --- 2. Split into chunks (roughly 1000 characters each) ---
chunk_size = 1000
chunks = textwrap.wrap(text, chunk_size)

print(f"✅ Created {len(chunks)} chunks")

# --- 3. Create embeddings ---
embeddings = []
for i, chunk in enumerate(chunks):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    ).data[0].embedding
    embeddings.append({"text": chunk, "embedding": emb, "file": input_file})
    print(f"Chunk {i+1}/{len(chunks)} embedded")

# --- 4. Save to JSON file ---
with open(output_file, "w") as f:
    json.dump(embeddings, f)

print(f"✅ Saved embeddings to {output_file}")

