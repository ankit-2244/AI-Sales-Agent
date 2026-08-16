from datetime import datetime, timezone


def qualify_lead(
    name: str,
    company: str,
    use_case: str,
    budget: str = "unknown"
):
    """
    Qualify a potential sales lead.
    """

    score = 0

    # Company provided
    if company:
        score += 25

    # Use case provided
    if use_case:
        score += 25

    # Budget provided
    if budget and budget.lower() != "unknown":
        score += 25

    # Name provided
    if name:
        score += 25

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