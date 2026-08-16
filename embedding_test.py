import os

import numpy as np
from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


texts = [
    "The agent should qualify leads.",
    "Can the system identify potential customers?",
    "What is the weather today?"
]


embeddings = []


# Create embeddings
for text in texts:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    embedding = response.embeddings[0].values

    embeddings.append(embedding)


# Convert to NumPy arrays
embedding_1 = np.array(embeddings[0])
embedding_2 = np.array(embeddings[1])
embedding_3 = np.array(embeddings[2])


# Calculate cosine similarity
similarity_12 = np.dot(embedding_1, embedding_2) / (
    np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
)

similarity_13 = np.dot(embedding_1, embedding_3) / (
    np.linalg.norm(embedding_1) * np.linalg.norm(embedding_3)
)


print("\nSimilarity between:")
print("1:", texts[0])
print("2:", texts[1])
print("Score:", similarity_12)


print("\nSimilarity between:")
print("1:", texts[0])
print("3:", texts[2])
print("Score:", similarity_13)