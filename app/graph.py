import re
import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.hybrid_search import hybrid_search
from app.reranker import rerank

from app.ingest import extract_url, ingest_site
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

    site_url: str
    site_context: str

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

        r"\bname\s*(?:is|:|-)?\s*([A-Za-z][A-Za-z.'-]{1,40})(?=\s*(?:,|$|email|e-?mail|and\b|company|budget|use\s*case|from|at))",
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

        r"\bbudg(?:et|jet)\s*(?:is|of|:)?\s*(\$?\s?[\d,]+(?:\.\d+)?\s*(?:usd|[kKmM])?)",

        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*usd",

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

        r"\buse\s*case\s*(?:is|:|-)?\s*([^.\n]+)",

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


GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "start",
    "let's start",
    "lets start",
}

INTAKE_FIELDS = (
    "name",
    "email",
    "company",
    "use_case",
    "budget",
)


def is_greeting(message: str) -> bool:
    text = clean_value(message).lower()
    return text in GREETINGS or text.rstrip("!.") in GREETINGS


def is_knowledge_question(message: str) -> bool:
    text = message.strip().lower()
    if not text:
        return False
    if text.endswith("?"):
        return True
    starters = (
        "what ",
        "how ",
        "does ",
        "do you",
        "can you",
        "is there",
        "tell me",
        "who ",
        "why ",
        "where ",
    )
    return text.startswith(starters)


FIELD_PREFIX = re.compile(
    r"^(?:sry[,]?\s+|sorry[,]?\s+|the\s+)*"
    r"(?:my\s+)?(?:name|email|e-?mail|company|use\s*case|budget|budjet)"
    r"\s*(?:is|:|-)?\s*",
    re.IGNORECASE,
)


def strip_field_prefix(message: str) -> str:
    return FIELD_PREFIX.sub("", message.strip(), count=1).strip(" :,-")


def looks_like_name(message: str) -> bool:
    text = strip_field_prefix(clean_value(message))
    if not text or is_greeting(text) or is_knowledge_question(text):
        return False
    if text.lower() in {"name", "email", "company", "budget", "use", "case"}:
        return False
    if "@" in text or "$" in text or "http" in text.lower() or any(ch.isdigit() for ch in text):
        return False
    words = text.split()
    if not 1 <= len(words) <= 3:
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", word) for word in words)


def missing_fields(values: dict, keys: tuple[str, ...]) -> list[str]:
    labels = {
        "name": "your name",
        "email": "your email address",
        "company": "your company",
        "use_case": "your use case",
        "budget": "your budget",
        "preferred_time": "your preferred time",
    }
    return [labels[key] for key in keys if not values.get(key)]


def join_missing(missing: list[str]) -> str:
    if len(missing) == 1:
        return missing[0]
    if len(missing) == 2:
        return missing[0] + " and " + missing[1]
    return ", ".join(missing[:-1]) + " and " + missing[-1]


COMPANY_HINTS = (
    "technologies",
    "technology",
    "tech",
    "inc",
    "ltd",
    "llc",
    "corp",
    "labs",
    "systems",
    "software",
    "solutions",
    "company",
)


def looks_like_company(message: str) -> bool:
    words = strip_field_prefix(clean_value(message)).lower().split()
    return any(word.rstrip(".") in COMPANY_HINTS for word in words)


def apply_labeled_parts(message: str, values: dict) -> dict:
    """Only set a field when the user labeled it (name / email / company / …)."""
    parts = re.split(r"\s*,\s*|\s+and\s+", message)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        email = extract_email(part)
        if email:
            values["email"] = email
        if re.search(r"\b(?:my\s+)?name\b", part, re.I):
            name = extract_name(part) or strip_field_prefix(part)
            if name and "@" not in name:
                values["name"] = clean_value(name)
        if re.search(r"\bcompany\b", part, re.I):
            company = extract_company(part) or strip_field_prefix(part)
            if company and "@" not in company:
                values["company"] = clean_value(company)
        if re.search(r"\buse\s*case\b", part, re.I):
            use_case = extract_use_case(part) or strip_field_prefix(part)
            if use_case:
                values["use_case"] = clean_value(use_case)
        if re.search(r"\bbudg(?:et|jet)\b", part, re.I):
            budget = extract_budget(part) or strip_field_prefix(part)
            if budget:
                values["budget"] = clean_value(budget)
    return values


def fill_bare_reply(message: str, values: dict) -> dict:
    """Unlabeled text goes into the next empty field. Never overwrite a filled field."""
    text = message.strip()
    if not text or is_greeting(text) or is_knowledge_question(text) or extract_url(text):
        return values

    values = apply_labeled_parts(text, values)

    email = extract_email(text)
    if email:
        values["email"] = email
        leftover = FIELD_PREFIX.sub("", text.replace(email, "")).strip(" ,;:-")
        if not leftover:
            return values

    if re.search(r"\b(name|email|e-?mail|company|use\s*case|budget|budjet)\b", text, re.I):
        return values

    if "@" in text:
        return values

    bare = strip_field_prefix(text)
    if not bare:
        return values

    for key in INTAKE_FIELDS:
        if values.get(key):
            continue
        if key == "name" and looks_like_company(bare):
            continue
        if key == "email":
            continue
        values[key] = clean_value(bare)
        break
    return values


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


    values = {
        "name": name,
        "email": email,
        "company": company,
        "use_case": use_case,
        "budget": budget,
        "preferred_time": preferred_time,
    }
    values = fill_bare_reply(message, values)
    name = values["name"]
    email = values["email"]
    company = values["company"]
    use_case = values["use_case"]
    budget = values["budget"]
    preferred_time = values["preferred_time"]

    # --------------------------------------------------------
    # INTENT ROUTING
    # --------------------------------------------------------

    wants_demo = (
        "book a demo" in text
        or "book demo" in text
        or "schedule a demo" in text
        or "schedule demo" in text
        or "book a meeting" in text
        or "schedule a meeting" in text
        or "want a demo" in text
        or ("demo" in text and ("book" in text or "schedule" in text))
    )
    wants_crm = (
        "add me to" in text
        or "add to the crm" in text
        or "add to crm" in text
        or "save my lead" in text
        or "save me" in text
        or "add my lead" in text
        or "create a crm" in text
    )
    wants_qualify = (
        "qualify" in text
        or "qualification" in text
        or "qualified lead" in text
        or "become a lead" in text
        or "qualify my lead" in text
        or "qualify this lead" in text
        or is_greeting(message)
    )

    found_url = extract_url(message)

    if found_url:
        intent = "ingest"
    elif wants_demo:
        intent = "book_demo"
    elif wants_crm:
        intent = "crm"
    elif wants_qualify:
        intent = "qualify"
    elif is_knowledge_question(message):
        intent = "knowledge"
    elif existing_intent in {"book_demo", "crm", "qualify"}:
        intent = existing_intent
    elif any(values[key] for key in INTAKE_FIELDS) and missing_fields(values, INTAKE_FIELDS):
        intent = "qualify"
    else:
        intent = "knowledge"

    out = {
        "intent": intent,
        "name": name,
        "email": email,
        "company": company,
        "use_case": use_case,
        "budget": budget,
        "preferred_time": preferred_time,
    }
    if found_url:
        out["site_url"] = found_url
    return out


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

        "site_url": decision.get("site_url") or state.get("site_url", ""),

    }


def _extractive_from_context(question: str, context: str) -> str:
    """Answer from retrieved/ingested text when Gemini is unavailable."""
    if not context.strip():
        return ""
    q_low = question.lower()
    pricing = any(w in q_low for w in ("price", "pricing", "cost", "plan", "how much", "$"))
    q_words = {
        w for w in re.findall(r"[a-z0-9$]+", q_low)
        if len(w) > 2 and w not in {"the", "and", "for", "you", "have", "what", "how", "hat"}
    }
    price_re = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d+\s*/\s*(?:user|month|mo)")
    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
    scored = []
    for sent in sentences:
        body = " ".join(sent.split()).strip()
        if len(body) < 24 or body.endswith("?"):
            continue
        low = body.lower()
        if "cookie" in low or "two paid plans, one free plan" in low:
            continue
        hits = sum(1 for w in q_words if w in low)
        if price_re.search(body):
            hits += 8
        if pricing and not price_re.search(body):
            continue
        if hits:
            scored.append((hits, body))
    scored.sort(reverse=True)
    quotes = [s for _, s in scored[:4]]
    if not quotes and pricing:
        money = [s for s in sentences if price_re.search(s or "")]
        quotes = [" ".join(s.split()) for s in money[:4] if len(s.strip()) > 20]
    if not quotes:
        return ""
    return (
        "I'm an AI sales assistant. From the ingested pages: "
        + " ".join(quotes)
    )


# ============================================================
# 6. RAG / KNOWLEDGE NODE
# ============================================================

def knowledge_node(
    state: SalesState
):

    question = state["customer_message"]
    documents = []
    try:
        hybrid_results = hybrid_search(question, top_k=5)
        documents = [document for document, score in hybrid_results]
        documents = rerank(question, documents, top_k=3)
    except Exception:
        documents = []

    context = "\n\n".join(documents)
    site_context = (state.get("site_context") or "").strip()
    if site_context:
        context = (
            "INGESTED COMPANY WEBSITE:\n"
            + site_context[:12000]
            + "\n\nINTERNAL KNOWLEDGE BASE:\n"
            + context
        )

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

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (response.text or "").strip()
        if text:
            return {"answer": text}
    except Exception as exc:
        fallback = _extractive_from_context(question, context)
        if fallback:
            return {"answer": fallback}
        err = str(exc)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return {
                "answer": (
                    "Gemini's free quota is used up for today, so I can't phrase a new answer. "
                    "Wait about a minute and try again, or ask tomorrow. "
                    "Your ingested site is still saved in this chat."
                )
            }
        return {"answer": "I couldn't reach the language model. Try again in a moment."}

    fallback = _extractive_from_context(question, context)
    return {"answer": fallback or "I don't have that in the ingested pages or knowledge base."}


# ============================================================
# 7. QUALIFICATION NODE
# ============================================================

def qualify_node(
    state: SalesState
):

    name = state.get("name", "")
    email = state.get("email", "")
    company = state.get("company", "")
    use_case = state.get("use_case", "")
    budget = state.get("budget", "")

    missing = missing_fields(
        {
            "name": name,
            "email": email,
            "company": company,
            "use_case": use_case,
            "budget": budget,
        },
        INTAKE_FIELDS,
    )

    if missing:
        if not name and not email and not company:
            return {
                "answer": (
                    "Hi — I'm the AI sales assistant. To get started, could you please provide "
                    + join_missing(missing)
                    + "?"
                )
            }
        thanks = f"Thanks {name}. " if name else "Thanks. "
        return {
            "answer": thanks + "Could you please provide " + join_missing(missing) + "?"
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

            f"{result['qualification_score']}"
            f" (budget on file: {budget})."

        )

    }


def ingest_node(state: SalesState):
    url = state.get("site_url") or extract_url(state.get("customer_message", ""))
    if not url:
        return {"answer": "I need a full website link starting with https://"}
    try:
        data = ingest_site(url, max_pages=6)
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"I could not ingest that site ({exc}). Try another URL."}
    if not data["page_count"]:
        return {
            "site_url": url,
            "site_context": "",
            "answer": (
                f"I reached {url} but found no usable HTML. "
                "JavaScript-heavy stores like Amazon often fail. Try a company marketing site."
            ),
        }
    titles = ", ".join(data["titles"][:4])
    return {
        "site_url": url,
        "site_context": data["context"],
        "answer": (
            f"Ingested {data['page_count']} pages from {url}. "
            f"I have: {titles}. Ask about plans, pricing, or products on that site."
        ),
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

    missing = missing_fields(
        {
            "name": state.get("name", ""),
            "email": state.get("email", ""),
            "company": state.get("company", ""),
        },
        ("name", "email", "company"),
    )
    if missing:
        return {
            "answer": (
                "I can add you to the CRM. Could you please provide "
                + join_missing(missing)
                + "?"
            )
        }

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

    if intent == "ingest":

        return "ingest"


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

builder.add_node(
    "ingest",
    ingest_node
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

        "ingest": "ingest",

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

builder.add_edge(
    "ingest",
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