import re
from datetime import datetime, timezone


def budget_amount(budget: str) -> float:
    if not budget:
        return 0.0
    text = budget.lower().replace(",", "").replace("usd", "").replace("$", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*([km])?", text)
    if not match:
        return 0.0
    value = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


def qualify_lead(
    name: str,
    company: str,
    use_case: str,
    budget: str = "unknown"
):
    """
    Completeness is 20 points each for name, company, and use case.
    Giving any budget is 15. Budget size adds up to 25 more.
    A tiny budget cannot reach 100.
    """

    score = 0

    if name:
        score += 20
    if company:
        score += 20
    if use_case:
        score += 20
    if budget and budget.lower() != "unknown":
        score += 15
        amount = budget_amount(budget)
        if amount >= 10_000:
            score += 25
        elif amount >= 1_000:
            score += 15
        elif amount > 0:
            score += 10

    if score >= 75:
        status = "qualified"

    elif score >= 50:
        status = "partially_qualified"

    else:
        status = "needs_more_information"

    return {
        "name": name,
        "company": company,
        "use_case": use_case,
        "budget": budget,
        "qualification_score": score,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

def book_demo(
    name: str,
    email: str,
    preferred_time: str
):
    """
    Mock demo-booking tool.

    In production this would connect to a
    calendar system such as Google Calendar.
    """

    return {
        "status": "booked",
        "name": name,
        "email": email,
        "preferred_time": preferred_time,
        "meeting_id": f"DEMO-{name.upper()}-001"
    }


def create_crm_lead(
    name: str,
    email: str,
    company: str,
    qualification_status: str,
    idempotency_key: str
):
    """
    Mock CRM lead creation.

    In production this would call an actual CRM API.
    """

    return {
        "status": "created",
        "crm_id": f"CRM-{idempotency_key}",
        "name": name,
        "email": email,
        "company": company,
        "qualification_status": qualification_status,
        "idempotency_key": idempotency_key
    }

if __name__ == "__main__":

    print("\n--- LEAD QUALIFICATION ---")

    lead = qualify_lead(
        name="Ankit",
        company="ABC Technologies",
        use_case="AI sales automation",
        budget="$10,000"
    )

    print(lead)


    print("\n--- DEMO BOOKING ---")

    demo = book_demo(
        name="Ankit",
        email="ankit@example.com",
        preferred_time="2026-08-15 11:00"
    )

    print(demo)


    print("\n--- CRM ---")

    crm = create_crm_lead(
        name="Ankit",
        email="ankit@example.com",
        company="ABC Technologies",
        qualification_status=lead["status"],
        idempotency_key="ANKIT-ABC-001"
    )

    print(crm)