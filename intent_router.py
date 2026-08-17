def is_claim_query(message: str) -> bool:
    claim_keywords = [
        "claim status",
        "check claim",
        "claim rejected",
        "documents required",
        "pre authorization",
        "claim approval",
        "claim settlement",
        "reimbursement"
    ]
    message = message.lower()
    return any(keyword in message for keyword in claim_keywords)
