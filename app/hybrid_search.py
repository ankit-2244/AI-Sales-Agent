import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not configured."
    )

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = (
    BASE_DIR / "data" / "knowledge_base.txt"
)

CHROMA_PATH = (
    BASE_DIR / "chroma_db"
)


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name="sales_knowledge"
)


# ============================================================
# 4. TOKENIZER
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# 5. CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


# ============================================================
# 6. INITIALIZE KNOWLEDGE BASE IF NEEDED
# ============================================================

def initialize_knowledge_base():

    # Check whether Chroma already contains documents.

    existing = collection.get(
        include=["documents"]
    )

    existing_documents = existing.get(
        "documents",
        []
    )

    if existing_documents:

        print(
            f"Chroma knowledge base already contains "
            f"{len(existing_documents)} documents."
        )

        return


    # --------------------------------------------------------
    # Load knowledge base
    # --------------------------------------------------------

    if not KNOWLEDGE_BASE_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge base not found at: "
            f"{KNOWLEDGE_BASE_PATH}"
        )

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        knowledge = file.read()


    # --------------------------------------------------------
    # Split into chunks
    # --------------------------------------------------------

    chunks = [
        chunk.strip()
        for chunk in knowledge.split("\n\n")
        if chunk.strip()
    ]


    if not chunks:

        raise RuntimeError(
            "Knowledge base is empty."
        )


    print(
        f"Creating embeddings for "
        f"{len(chunks)} knowledge chunks..."
    )


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    documents = []
    embeddings = []
    ids = []


    for index, chunk in enumerate(chunks):

        print(
            f"Embedding chunk "
            f"{index + 1}/{len(chunks)}..."
        )

        embedding = create_embedding(
            chunk
        )

        documents.append(
            chunk
        )

        embeddings.append(
            embedding
        )

        ids.append(
            f"chunk_{index}"
        )


    # --------------------------------------------------------
    # Store in Chroma
    # --------------------------------------------------------

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )


    print(
        f"Successfully indexed "
        f"{len(chunks)} knowledge chunks."
    )


# ============================================================
# 7. GET DOCUMENTS FOR BM25
# ============================================================

def get_documents():

    initialize_knowledge_base()

    stored_data = collection.get(
        include=["documents"]
    )

    documents = stored_data.get(
        "documents",
        []
    )

    if not documents:

        raise RuntimeError(
            "No documents found in Chroma collection."
        )

    return documents


# ============================================================
# 8. HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    top_k=5
):

    # Make sure the knowledge base exists.

    documents = get_documents()


    # --------------------------------------------------------
    # Prepare BM25
    # --------------------------------------------------------

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )


    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    query_embedding = create_embedding(
        query
    )

    vector_results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=min(
            top_k,
            len(documents)
        )
    )

    vector_documents = (
        vector_results["documents"][0]
        if vector_results.get("documents")
        else []
    )


    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    tokenized_query = tokenize(
        query
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )


    # --------------------------------------------------------
    # VECTOR RANKING
    # --------------------------------------------------------

    vector_rank = {}

    for rank, document in enumerate(
        vector_documents,
        start=1
    ):

        vector_rank[
            document
        ] = rank


    # --------------------------------------------------------
    # BM25 RANKING
    # --------------------------------------------------------

    bm25_rank = {}

    for rank, index in enumerate(
        bm25_ranked_indices,
        start=1
    ):

        bm25_rank[
            documents[index]
        ] = rank


    # --------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------------

    combined_scores = {}

    all_documents = (
        set(vector_rank.keys())
        |
        set(bm25_rank.keys())
    )


    for document in all_documents:

        score = 0


        if document in vector_rank:

            score += 1 / (
                60 + vector_rank[document]
            )


        if document in bm25_rank:

            score += 1 / (
                60 + bm25_rank[document]
            )


        combined_scores[
            document
        ] = score


    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    ranked_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    return ranked_results[:top_k]


# ============================================================
# 9. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "Customer: "
    )


    results = hybrid_search(
        question,
        top_k=5
    )


    print(
        "\nHybrid search results:\n"
    )


    for rank, (
        document,
        score
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Score: {score:.4f}"
        )

        print(
            document
        )

        print(
            "-" * 60
        )