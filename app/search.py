import os

import chromadb
from dotenv import load_dotenv
from google import genai


# Load API key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Connect to our existing Chroma database
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Get our collection
collection = chroma_client.get_collection(
    name="sales_knowledge"
)


# Ask the customer
question = input("Customer: ")


# Convert the question into an embedding
response = client.models.embed_content(
    model="gemini-embedding-2",
    contents=question
)

query_embedding = response.embeddings[0].values


# Search the vector database
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


# Display results
print("\nMost relevant information:\n")

for i, document in enumerate(results["documents"][0]):

    print(f"Result {i + 1}:")
    print(document)
    print("-" * 60)