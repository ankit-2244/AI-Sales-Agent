import os

import chromadb
from dotenv import load_dotenv
from google import genai


# Load API key from .env
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Load knowledge base
with open(
    "data/knowledge_base.txt",
    "r",
    encoding="utf-8"
) as file:

    knowledge = file.read()


# Split knowledge into chunks
chunks = [
    chunk.strip()
    for chunk in knowledge.split("\n\n")
    if chunk.strip()
]


# Create persistent Chroma database
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Create collection
collection = chroma_client.get_or_create_collection(
    name="sales_knowledge"
)


# Create embeddings and store them
documents = []
embeddings = []
ids = []


for index, chunk in enumerate(chunks):

    print(f"Creating embedding {index + 1}/{len(chunks)}...")

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk
    )

    embedding = response.embeddings[0].values

    documents.append(chunk)
    embeddings.append(embedding)
    ids.append(f"chunk_{index}")


# Save everything into Chroma
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)


print()
print(f"Successfully indexed {len(chunks)} knowledge chunks.")
print("Vector database created in: ./chroma_db")