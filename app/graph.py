import re
import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.hybrid_search import hybrid_search
from app.reranker import rerank

from app.tools import (
    qualify_lead,
    book_demo,
    create_crm_lead,
)


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# 2. STATE
# ============================================================

class SalesState(TypedDict, total=False):

    customer_message: str

    intent: str

    name: str
    email: str
    company: str
    use_case: str
    budget: str
    preferred_time: str

    result: dict
    answer: str


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_value(value: str) -> str:
    """
    Clean extracted values.
    """

    if not value:
        return ""

    value = value.strip()

    value = value.strip(
        " \t\n\r.,;:!?-"
    )

    return value


def extract_name(message: str) -> str:
    """
    Extract names from examples such as:

    My name is Ankit
    I am Ankit
    I'm Ankit
    Name: Ankit
    Name is Ankit
    """

    patterns = [

        r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z .'-]{0,50}?)(?=\s+(?:from|at|of|and|works?|email|budget)\b|[,.!?]|$)",

        r"\bi\s+am\s+([A-Za-z][A-Za-z .'-]{0,50}?)(?=\s+(?:from|at|of|and|email|budget)\b|[,.!?]|$)",

        r"\bi['’]m\s+([A-Za-z][A-Za-z .'-]{0,50}?)(?=\s+(?:from|at|of|and|email|budget)\b|[,.!?]|$)",

        r"\bname\s*(?:is|:)\s*([A-Za-z][A-Za-z .'-]{0,50}?)(?=\s+(?:from|at|of|and|email|budget)\b|[,.!?]|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            return clean_value(
                match.group(1)
            )

    return ""


def extract_email(message: str) -> str:
    """
    Extract email address.
    """

    match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        message
    )

    if match:
        return match.group(0)

    return ""


def extract_company(message: str) -> str:
    """
    Extract company names from examples such as:

    Company: Acme Technologies
    Company is Acme Technologies
    I work at Acme Technologies
    I work for Acme Technologies
    I am Ankit from Acme Technologies
    From Acme Technologies
    """

    patterns = [

        # Company: Acme Technologies
        r"\bcompany\s*:\s*([^.,;\n]+)",

        # Company is Acme Technologies
        r"\bcompany\s+is\s+([^.,;\n]+)",

        # I work at Acme Technologies
        r"\bi\s+work\s+at\s+([^.,;\n]+)",

        # I work for Acme Technologies
        r"\bi\s+work\s+for\s+([^.,;\n]+)",

        # I am Ankit from Acme Technologies
        r"\bfrom\s+([^.,;\n]+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            company = clean_value(
                match.group(1)
            )

            # Remove things that are clearly not part
            # of the company name.
            company = re.split(
                r"\b(?:and|with|our|we|my|budget|use\s+case|sales\s+representatives?|expected|start)\b",
                company,
                maxsplit=1,
                flags=re.IGNORECASE
            )[0]

            company = clean_value(company)

            if company:
                return company

    return ""


def extract_budget(message: str) -> str:
    """
    Extract budgets such as:

    Budget: $50,000
    Budget is $50,000
    Our budget is $10,000
    Budget of $50k
    $50,000
    """

    patterns = [

        r"\bbudget\s*(?:is|of|:)?\s*(\$?\s?[\d,]+(?:\.\d+)?\s*[kKmM]?)",

        r"\$\s?[\d,]+(?:\.\d+)?\s*[kKmM]?",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            if match.lastindex:
                return clean_value(
                    match.group(1)
                )

            return clean_value(
                match.group(0)
            )

    return ""


def extract_use_case(message: str) -> str:
    """
    Extract use cases from examples such as:

    We want AI sales automation.
    We need AI sales automation.
    Use case: AI sales automation.
    Our use case is AI sales automation.
    We want an AI sales agent to qualify leads.
    """

    patterns = [

        r"\buse\s+case\s*:\s*([^.\n]+)",

        r"\buse\s+case\s+is\s+([^.\n]+)",

        r"\bour\s+use\s+case\s+is\s+([^.\n]+)",

        r"\bwe\s+want\s+([^.\n]+)",

        r"\bwe\s+need\s+([^.\n]+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            value = clean_value(
                match.group(1)
            )

            if value:
                return value

    return ""


def extract_preferred_time(message: str) -> str:
    """
    Extract demo time.

    Examples:

    tomorrow at 11 AM
    tomorrow 11 AM
    today at 3 PM
    11 AM
    14:30 PM
    """

    patterns = [

        r"\b(tomorrow(?:\s+at)?\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))",

        r"\b(today(?:\s+at)?\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))",

        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            return clean_value(
                match.group(1)
            )

    return ""


# ============================================================
# 4. PLANNER
# ============================================================

def planner(state: SalesState):

    message = state.get(
        "customer_message",
        ""
    ).strip()

    text = message.lower()

    # --------------------------------------------------------
    # Existing state
    # --------------------------------------------------------

    existing_intent = state.get(
        "intent",
        ""
    )

    name = state.get(
        "name",
        ""
    )

    email = state.get(
        "email",
        ""
    )

    company = state.get(
        "company",
        ""
    )

    use_case = state.get(
        "use_case",
        ""
    )

    budget = state.get(
        "budget",
        ""
    )

    preferred_time = state.get(
        "preferred_time",
        ""
    )


    # --------------------------------------------------------
    # Extract information from current message
    # --------------------------------------------------------

    extracted_name = extract_name(
        message
    )

    if extracted_name:
        name = extracted_name


    extracted_email = extract_email(
        message
    )

    if extracted_email:
        email = extracted_email


    extracted_company = extract_company(
        message
    )

    if extracted_company:
        company = extracted_company


    extracted_budget = extract_budget(
        message
    )

    if extracted_budget:
        budget = extracted_budget


    extracted_use_case = extract_use_case(
        message
    )

    if extracted_use_case:
        use_case = extracted_use_case


    extracted_time = extract_preferred_time(
        message
    )

    if extracted_time:
        preferred_time = extracted_time


    # --------------------------------------------------------
    # INTENT ROUTING
    # --------------------------------------------------------

    # 1. DEMO BOOKING
    if (
        "book a demo" in text
        or "book demo" in text
        or "schedule a demo" in text
        or "schedule demo" in text
        or "book a meeting" in text
        or "schedule a meeting" in text
        or "want a demo" in text
        or "demo" in text and (
            "book" in text
            or "schedule" in text
        )
    ):

        intent = "book_demo"


    # Continue existing demo conversation
    elif existing_intent == "book_demo":

        intent = "book_demo"


    # 2. CRM
    elif (
        "crm" in text
        or "customer relationship management" in text
        or "add me to" in text
        or "save my lead" in text
        or "save me" in text
        or "add my lead" in text
    ):

        intent = "crm"


    # Continue existing CRM conversation
    elif existing_intent == "crm":

        intent = "crm"


    # 3. QUALIFICATION
    elif (
        "qualify" in text
        or "qualification" in text
        or "qualified lead" in text
        or "become a lead" in text
        or "qualify my lead" in text
        or "qualify this lead" in text
    ):

        intent = "qualify"


    # Continue existing qualification conversation
    elif existing_intent == "qualify":

        intent = "qualify"


    # If multiple lead fields are supplied,
    # infer that this is a qualification request.
    elif sum(
        bool(value)
        for value in [
            name,
            company,
            use_case,
            budget
        ]
    ) >= 2:

        intent = "qualify"


    # Otherwise RAG / knowledge question
    else:

        intent = "knowledge"


    # --------------------------------------------------------
    # Return planner decision
    # --------------------------------------------------------

    return {

        "intent": intent,

        "name": name,

        "email": email,

        "company": company,

        "use_case": use_case,

        "budget": budget,

        "preferred_time": preferred_time,

    }


# ============================================================
# 5. PLANNER NODE
# ============================================================

def planner_node(
    state: SalesState
):

    decision = planner(
        state
    )

    return {

        "intent": decision.get(
            "intent",
            "knowledge"
        ),

        "name": decision.get(
            "name",
            ""
        ),

        "email": decision.get(
            "email",
            ""
        ),

        "company": decision.get(
            "company",
            ""
        ),

        "use_case": decision.get(
            "use_case",
            ""
        ),

        "budget": decision.get(
            "budget",
            ""
        ),

        "preferred_time": decision.get(
            "preferred_time",
            ""
        ),

    }


# ============================================================
# 6. RAG / KNOWLEDGE NODE
# ============================================================

def knowledge_node(
    state: SalesState
):

    question = state[
        "customer_message"
    ]


    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    hybrid_results = hybrid_search(
        question,
        top_k=5
    )


    documents = [

        document

        for document, score

        in hybrid_results

    ]


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    relevant_documents = rerank(
        question,
        documents,
        top_k=3
    )


    context = "\n\n".join(
        relevant_documents
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    prompt = f"""
You are an AI Sales Agent.

Answer the customer using ONLY the company
information provided below.

Never invent:

- prices
- discounts
- capabilities
- integrations
- delivery dates
- commercial terms

If the information is unavailable, say so.

COMPANY INFORMATION:

{context}

CUSTOMER:

{question}

Give a concise and helpful answer.
"""


    response = client.models.generate_content(

        model="gemini-2.5-flash",

        contents=prompt

    )


    return {

        "answer": response.text

    }


# ============================================================
# 7. QUALIFICATION NODE
# ============================================================

def qualify_node(
    state: SalesState
):

    name = state.get(
        "name",
        ""
    )

    company = state.get(
        "company",
        ""
    )

    use_case = state.get(
        "use_case",
        ""
    )

    budget = state.get(
        "budget",
        ""
    )


    # --------------------------------------------------------
    # Find missing information
    # --------------------------------------------------------

    missing = []


    if not name:

        missing.append(
            "your name"
        )


    if not company:

        missing.append(
            "your company"
        )


    if not use_case:

        missing.append(
            "your use case"
        )


    if not budget:

        missing.append(
            "your budget"
        )


    # --------------------------------------------------------
    # Ask only for missing information
    # --------------------------------------------------------

    if missing:

        if len(missing) == 1:

            question = missing[0]

        elif len(missing) == 2:

            question = (
                missing[0]
                + " and "
                + missing[1]
            )

        else:

            question = (
                ", ".join(
                    missing[:-1]
                )
                + " and "
                + missing[-1]
            )


        return {

            "answer": (
                "I'd be happy to qualify "
                "your lead. Could you please "
                "provide "
                + question
                + "?"
            )

        }


    # --------------------------------------------------------
    # All information available
    # --------------------------------------------------------

    result = qualify_lead(

        name=name,

        company=company,

        use_case=use_case,

        budget=budget,

    )


    return {

        "result": result,

        "answer": (

            "Thanks for the information. "

            "Your lead has been classified as "

            f"'{result['status']}' "

            "with a qualification score of "

            f"{result['qualification_score']}."

        )

    }


# ============================================================
# 8. DEMO NODE
# ============================================================

def demo_node(
    state: SalesState
):

    name = state.get(
        "name",
        ""
    )

    email = state.get(
        "email",
        ""
    )

    preferred_time = state.get(
        "preferred_time",
        ""
    )


    # --------------------------------------------------------
    # Find missing demo information
    # --------------------------------------------------------

    missing = []


    if not name:

        missing.append(
            "your name"
        )


    if not email:

        missing.append(
            "your email address"
        )


    if not preferred_time:

        missing.append(
            "your preferred time"
        )


    # --------------------------------------------------------
    # Ask for missing information
    # --------------------------------------------------------

    if missing:

        if len(missing) == 1:

            question = missing[0]

        elif len(missing) == 2:

            question = (
                missing[0]
                + " and "
                + missing[1]
            )

        else:

            question = (
                ", ".join(
                    missing[:-1]
                )
                + " and "
                + missing[-1]
            )


        return {

            "answer": (

                "Sure, I can help you "
                "book a demo. Could you "
                "please provide "
                + question
                + "?"

            )

        }


    # --------------------------------------------------------
    # Book demo
    # --------------------------------------------------------

    result = book_demo(

        name=name,

        email=email,

        preferred_time=preferred_time,

    )


    return {

        "result": result,

        "answer": (

            "Your demo has been booked "
            "successfully. Your meeting ID "
            f"is {result['meeting_id']}."

        )

    }


# ============================================================
# 9. CRM NODE
# ============================================================

def crm_node(
    state: SalesState
):

    # --------------------------------------------------------
    # Create deterministic idempotency key
    # --------------------------------------------------------

    key = (

        f"{state.get('name', 'unknown')}-"

        f"{state.get('company', 'unknown')}"

    ).upper().replace(
        " ",
        "-"
    )


    # --------------------------------------------------------
    # Create CRM lead
    # --------------------------------------------------------

    result = create_crm_lead(

        name=state.get(
            "name",
            ""
        ),

        email=state.get(
            "email",
            ""
        ),

        company=state.get(
            "company",
            ""
        ),

        qualification_status="qualified",

        idempotency_key=key,

    )


    return {

        "result": result,

        "answer": (

            "Your lead has been added "
            "to our CRM. CRM ID: "

            f"{result['crm_id']}."

        )

    }


# ============================================================
# 10. ROUTER
# ============================================================

def route_after_planner(
    state: SalesState
):

    intent = state.get(
        "intent",
        "knowledge"
    )


    if intent == "qualify":

        return "qualify"


    if intent == "book_demo":

        return "book_demo"


    if intent == "crm":

        return "crm"


    return "knowledge"


# ============================================================
# 11. BUILD GRAPH
# ============================================================

builder = StateGraph(
    SalesState
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "knowledge",
    knowledge_node
)

builder.add_node(
    "qualify",
    qualify_node
)

builder.add_node(
    "book_demo",
    demo_node
)

builder.add_node(
    "crm",
    crm_node
)


# ------------------------------------------------------------
# Starting point
# ------------------------------------------------------------

builder.add_edge(
    START,
    "planner"
)


# ------------------------------------------------------------
# Planner routing
# ------------------------------------------------------------

builder.add_conditional_edges(

    "planner",

    route_after_planner,

    {

        "knowledge": "knowledge",

        "qualify": "qualify",

        "book_demo": "book_demo",

        "crm": "crm",

    }

)


# ------------------------------------------------------------
# End points
# ------------------------------------------------------------

builder.add_edge(
    "knowledge",
    END
)

builder.add_edge(
    "qualify",
    END
)

builder.add_edge(
    "book_demo",
    END
)

builder.add_edge(
    "crm",
    END
)


# ============================================================
# 12. MEMORY / CHECKPOINTING
# ============================================================

memory = InMemorySaver()


sales_graph = builder.compile(
    checkpointer=memory
)


# ============================================================
# 13. LOCAL TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Sales Agent started."
    )

    print(
        "Type 'exit' to quit.\n"
    )


    thread_id = "demo-user-1"


    config = {

        "configurable": {

            "thread_id": thread_id

        }

    }


    while True:

        question = input(
            "Customer: "
        )


        if question.lower() == "exit":

            break


        result = sales_graph.invoke(

            {

                "customer_message":
                question

            },

            config=config

        )


        print(
            "\nAgent:"
        )

        print(

            result.get(

                "answer",

                "I couldn't process "
                "that request."

            )

        )

        print()