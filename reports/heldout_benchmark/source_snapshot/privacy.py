import re


def redact(text: str) -> str:
    text = re.sub(r'\bsk-[A-Za-z0-9_-]{15,}\b', '[SECRET]', text)
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'(?<!\w)@[A-Za-z0-9_]+', '[USER]', text)
    text = re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)', '[NUMBER]', text)
    return re.sub(r'\s+', ' ', text).strip()


def risk_signals(text: str) -> list[str]:
    patterns = {
        'account_compromise': r'\b(hacked|account.{0,12}stolen|identity theft|unauthori[sz]ed (?:access|login))\b',
        'legal_threat': r'\b(lawsuit|my lawyer|sue you|legal action)\b',
        'physical_safety': r'\b(injured|injury|caught fire|on fire|unsafe driver|assault|hurt me)\b',
        'instruction_injection': r'\b(ignore (?:all |your |previous )?instructions|reveal (?:the |your )?system prompt)\b',
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]


def validate_reply(reply: str, policy_ids: list[str], case_ids: list[str], policies: list[dict], cases: list[dict]) -> list[str]:
    issues = []
    if not reply.strip():
        issues.append('Empty reply')
    if len(reply) > 280:
        issues.append('Reply exceeds 280 Unicode characters')
    if re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)', reply):
        issues.append('Reply contains a personal identifier')
    if re.search(r'\b(?:send|share|provide|DM|tell).{0,35}\b(?:password|one.time code|OTP|verification code|credit card|CVV)\b', reply, re.I):
        issues.append('Reply requests a credential or payment secret')
    if re.search(r'\b(?:send|share|provide|post|tell).{0,25}\b(?:address|phone number|email|order number|account number)\b', reply, re.I) and not re.search(r'\b(?:DM|private|secure|in.app)\b', reply, re.I):
        issues.append('Reply requests private information publicly')
    if re.search(r"\b(?:we|I)(?: have|'ve)?\s+(?:already\s+)?(?:refunded|issued|processed|credited|cancelled|canceled|reset|booked)\b|\bguarantee\b|\bwe will refund\b|\bwill (?:arrive|be delivered) (?:tomorrow|today|within)\b", reply, re.I):
        issues.append('Reply promises an unverified account action or outcome')
    if '[URL]' in reply or '[USER]' in reply or '[EMAIL]' in reply or '[NUMBER]' in reply:
        issues.append('Reply contains a historical-data placeholder')
    valid_policies = {p['policy_id'] for p in policies}
    valid_cases = {c['case_id'] for c in cases}
    if not set(policy_ids).issubset(valid_policies) or not set(case_ids).issubset(valid_cases):
        issues.append('Reply cites evidence that was not retrieved')
    allowed_urls = {p['source_url'].rstrip('/') for p in policies}
    for url in re.findall(r'https?://[^\s<>]+', reply):
        if url.rstrip('.,!/)') not in {u.rstrip(')') for u in allowed_urls}:
            issues.append('Reply contains a URL outside retrieved policy sources')
    return list(dict.fromkeys(issues))
