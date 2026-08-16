import json
import os

from dotenv import load_dotenv
from google import genai

from app.hybrid_search import hybrid_search


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def rerank(query, documents, top_k=3):

    candidates = []

    for index, document in enumerate(documents):

        candidates.append({
            "id": index,
            "text": document
        })

    prompt = f"""
You are a retrieval reranker.

Your job is NOT to answer the customer.

Your job is to rank the provided documents
according to how useful they are for answering
the customer's question.

CUSTOMER QUESTION:
{query}

DOCUMENTS:
{json.dumps(candidates, indent=2)}

Return ONLY valid JSON in this format:

{{
    "ranked_ids": [0, 1, 2]
}}

Put the most relevant document first.

Do not invent information.
Do not explain your answer.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    result = json.loads(raw)

    ranked_ids = result["ranked_ids"]

    ranked_documents = [
        candidates[index]["text"]
        for index in ranked_ids
        if 0 <= index < len(candidates)
    ]

    return ranked_documents[:top_k]


if __name__ == "__main__":

    question = input("Customer: ")

    # Get hybrid candidates
    hybrid_results = hybrid_search(
        question,
        top_k=5
    )

    documents = [
        document
        for document, score in hybrid_results
    ]

    # Rerank
    final_results = rerank(
        question,
        documents,
        top_k=3
    )

    print("\nReranked results:\n")

    for rank, document in enumerate(
        final_results,
        start=1
    ):

        print(f"Rank {rank}:")
        print(document)
        print("-" * 60)