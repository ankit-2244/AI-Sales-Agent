import os

import numpy as np
from dotenv import load_dotenv
from google import genai


# -----------------------------
# 1. Load API key
# -----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# -----------------------------
# 2. Load knowledge base
# -----------------------------

with open("knowledge_base.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()


# -----------------------------
# 3. Split knowledge into chunks
# -----------------------------

chunks = knowledge.split("\n\n")


# -----------------------------
# 4. Function to create embedding
# -----------------------------

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(response.embeddings[0].values)


# -----------------------------
# 5. Create embeddings for chunks
# -----------------------------

chunk_embeddings = []

for chunk in chunks:

    embedding = create_embedding(chunk)

    chunk_embeddings.append(embedding)


# -----------------------------
# 6. Ask customer a question
# -----------------------------

customer_message = input("Customer: ")


# -----------------------------
# 7. Create embedding for question
# -----------------------------

question_embedding = create_embedding(customer_message)


# -----------------------------
# 8. Calculate similarity
# -----------------------------

scores = []

for chunk, chunk_embedding in zip(chunks, chunk_embeddings):

    similarity = np.dot(question_embedding, chunk_embedding) / (
        np.linalg.norm(question_embedding)
        * np.linalg.norm(chunk_embedding)
    )

    scores.append((similarity, chunk))


# -----------------------------
# 9. Sort by highest similarity
# -----------------------------

scores.sort(reverse=True, key=lambda x: x[0])


# -----------------------------
# 10. Display results
# -----------------------------

print("\nMost relevant information:\n")

for score, chunk in scores[:3]:

    print("Similarity:", round(score, 3))
    print(chunk)
    print("-" * 50)