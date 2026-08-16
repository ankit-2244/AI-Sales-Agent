import os

from dotenv import load_dotenv
from google import genai

from app.hybrid_search import hybrid_search
from app.reranker import rerank


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def sales_agent(customer_message: str):

    # --------------------------------
    # 1. Hybrid retrieval
    # --------------------------------

    hybrid_results = hybrid_search(
        customer_message,
        top_k=5
    )

    documents = [
        document
        for document, score in hybrid_results
    ]


    # --------------------------------
    # 2. Reranking
    # --------------------------------

    relevant_documents = rerank(
        customer_message,
        documents,
        top_k=3
    )


    # --------------------------------
    # 3. Build grounded context
    # --------------------------------

    context = "\n\n".join(
        relevant_documents
    )


    # --------------------------------
    # 4. Generate final answer
    # --------------------------------

    prompt = f"""
You are an AI Sales Agent.

You must answer the customer using ONLY
the company information provided below.

IMPORTANT RULES:

1. You are an AI. Never pretend to be human.

2. Never invent information.

3. Never invent:
   - prices
   - discounts
   - delivery dates
   - product capabilities
   - integrations
   - commercial terms

4. If the information below does not answer
the customer's question, say that the information
is not available.

5. Be concise and helpful.

COMPANY INFORMATION:
{context}

CUSTOMER QUESTION:
{customer_message}

Answer the customer now.
"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    return response.text


if __name__ == "__main__":

    question = input(
        "Customer: "
    )

    answer = sales_agent(
        question
    )

    print(
        "\nAgent:"
    )

    print(answer)