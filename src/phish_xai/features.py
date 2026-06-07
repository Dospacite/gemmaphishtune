from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup


SUSPICIOUS_URL_TERMS = {
    "account",
    "auth",
    "bank",
    "billing",
    "confirm",
    "credential",
    "login",
    "password",
    "pay",
    "payment",
    "secure",
    "signin",
    "update",
    "verify",
    "wallet",
    "webscr",
}

SOCIAL_ENGINEERING_TERMS = {
    "account suspended",
    "act now",
    "billing information",
    "confirm your",
    "credit card",
    "limited time",
    "login",
    "password",
    "payment failed",
    "security alert",
    "sign in",
    "unusual activity",
    "verify",
    "wallet",
}

SYSTEM_PROMPT = (
    "You are a security classifier for website phishing detection. "
    "Classify websites using only the supplied URL, captured page text, page-structure counts, "
    "and HTTP fetch metadata. Return compact JSON with keys label, confidence, and explanation. "
    "The label must be exactly one of: phishing, legitimate. "
    "Do not use domain registration, WHOIS/RDAP, reputation feeds, or human review."
)


@dataclass(frozen=True)
class WebsiteFeatures:
    record_id: str
    url: str
    normalized_url: str
    final_url: str
    title: str
    text: str
    link_count: int
    external_resource_count: int
    forms_count: int
    password_fields: int
    input_fields: int
    iframe_count: int
    script_count: int
    status_code: int | None
    redirect_count: int
    url_signals: list[str]
    content_signals: list[str]


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower().strip()
    path = parsed.path or "/"
    return f"{scheme}://{netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


def compact_text(text: str, max_chars: int) -> str:
    text = html_lib.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def clean_html_text(raw_html: str, max_chars: int) -> str:
    soup = BeautifulSoup(raw_html or "", "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    return compact_text(soup.get_text(" "), max_chars=max_chars)


def count_links_and_resources(soup: BeautifulSoup) -> tuple[int, int]:
    link_count = 0
    resource_count = 0
    for tag, attr in (("a", "href"), ("form", "action")):
        for node in soup.find_all(tag):
            raw = node.get(attr)
            if not isinstance(raw, str) or not raw.strip() or raw.strip().startswith(("#", "javascript:")):
                continue
            link_count += 1
    for tag, attr in (("script", "src"), ("link", "href"), ("iframe", "src")):
        for node in soup.find_all(tag):
            raw = node.get(attr)
            if not isinstance(raw, str) or not raw.strip() or raw.strip().startswith(("#", "javascript:")):
                continue
            resource_count += 1
    return link_count, resource_count


def suspicious_url_signals(url: str) -> list[str]:
    signals: list[str] = []
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    path_query = f"{parsed.path}?{parsed.query}".lower()

    if parsed.scheme != "https":
        signals.append("URL does not use HTTPS.")
    if len(normalized) >= 90:
        signals.append(f"URL is long ({len(normalized)} characters).")
    matched_terms = sorted(term for term in SUSPICIOUS_URL_TERMS if term in path_query)
    if matched_terms:
        signals.append("URL path or query contains sensitive-action terms: " + ", ".join(matched_terms[:6]) + ".")
    if "@" in normalized:
        signals.append("URL contains an @ character, which can obscure the visible destination.")
    if normalized.count("-") >= 4:
        signals.append("URL contains many hyphen separators.")
    if not signals:
        signals.append("URL does not show strong lexical phishing indicators.")
    return signals


def content_signals(soup: BeautifulSoup, text: str, link_count: int, resource_count: int) -> list[str]:
    signals: list[str] = []
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    password_inputs = soup.find_all("input", {"type": re.compile("^password$", re.I)})
    if password_inputs:
        signals.append(f"Page contains {len(password_inputs)} password input field(s).")
    elif forms:
        signals.append(f"Page contains {len(forms)} form(s) but no password field.")
    else:
        signals.append("No HTML forms were detected in the captured page.")

    lowered = text.lower()
    matched = sorted(term for term in SOCIAL_ENGINEERING_TERMS if term in lowered)
    if matched:
        signals.append("Page text contains social-engineering terms: " + ", ".join(matched[:6]) + ".")

    if link_count or resource_count:
        signals.append(f"Page contains {link_count} link/form target(s) and {resource_count} script/link/iframe resource reference(s).")
    else:
        signals.append("No page links, form targets, or script/link/iframe resources were detected.")

    if len(text) < 120:
        signals.append("Captured body text is very short, which limits confidence.")
    return signals


def extract_website_features(
    doc: dict[str, Any],
    max_text_chars: int = 3500,
) -> WebsiteFeatures:
    url = str(doc.get("url") or (doc.get("metadata") or {}).get("url") or "")
    normalized = normalize_url(url)
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    final_url = str(metadata.get("final_url") or metadata.get("url") or url)
    html = str(doc.get("html") or "")
    soup = BeautifulSoup(html, "lxml")
    soup_title = soup.title.string if soup.title and soup.title.string else ""
    title = compact_text(str(doc.get("title") or soup_title or ""), 180)
    text = clean_html_text(html, max_chars=max_text_chars)
    link_count, resource_count = count_links_and_resources(soup)
    url_sigs = suspicious_url_signals(final_url or url)
    content = content_signals(soup, text, link_count, resource_count)

    status_code = metadata.get("status_code")
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    redirect_count = metadata.get("redirect_count")
    try:
        redirect_count = int(redirect_count or 0)
    except (TypeError, ValueError):
        redirect_count = 0

    if redirect_count >= 3:
        content.append(f"Request followed {redirect_count} redirects before landing on the final URL.")
    if status_code is not None:
        content.append(f"HTTP status code was {status_code}.")

    forms_count = len(soup.find_all("form"))
    password_fields = len(soup.find_all("input", {"type": re.compile("^password$", re.I)}))
    input_fields = len(soup.find_all("input"))
    iframe_count = len(soup.find_all("iframe"))
    script_count = len(soup.find_all("script"))

    return WebsiteFeatures(
        record_id=stable_id(normalized),
        url=url,
        normalized_url=normalized,
        final_url=final_url,
        title=title,
        text=text,
        link_count=link_count,
        external_resource_count=resource_count,
        forms_count=forms_count,
        password_fields=password_fields,
        input_fields=input_fields,
        iframe_count=iframe_count,
        script_count=script_count,
        status_code=status_code,
        redirect_count=redirect_count,
        url_signals=url_sigs,
        content_signals=content,
    )


def format_input_features(features: WebsiteFeatures) -> str:
    title = features.title or "No title captured."
    page_stats = [
        f"links_or_form_targets: {features.link_count}",
        f"script_link_iframe_resources: {features.external_resource_count}",
        f"forms: {features.forms_count}",
        f"password_fields: {features.password_fields}",
        f"input_fields: {features.input_fields}",
        f"iframes: {features.iframe_count}",
        f"scripts: {features.script_count}",
    ]
    if features.status_code is not None:
        page_stats.append(f"http_status_code: {features.status_code}")
    page_stats.append(f"redirect_count: {features.redirect_count}")
    return "\n".join(
        [
            "# Information:",
            "## URL:",
            features.final_url or features.url,
            "## Title:",
            title,
            "## Content:",
            features.text or "No page body text captured.",
            "## Page Structure:",
            "\n".join(f"- {item}" for item in page_stats),
        ]
    )


def automatic_explanation(features: WebsiteFeatures, label: str) -> str:
    verdict = "phishing" if label == "phishing" else "legitimate"
    sections = [
        f"Predicted as {verdict} based on automatically extracted website evidence.",
        "URL signals: " + " ".join(features.url_signals[:4]),
        "Website content and link signals: " + " ".join(features.content_signals[:5]),
    ]
    if label == "legitimate":
        sections.append(
            "Legitimate examples are drawn from the Tranco corpus and still require deployment-time monitoring because popularity is not a proof of safety."
        )
    return "\n".join(sections)


def format_prompt(input_features: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{format_user_message(input_features)}"


def format_user_message(input_features: str) -> str:
    return (
        "Classify the following website evidence using only the fields shown below.\n\n"
        f"{input_features}\n"
        "# Pred:\n"
    )


def format_messages(input_features: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_message(input_features).rstrip()},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def format_answer(label: str, explanation: str, confidence: float = 0.75) -> str:
    return json.dumps(
        {
            "label": label,
            "confidence": round(float(confidence), 3),
            "explanation": explanation,
        },
        ensure_ascii=False,
    )


def training_text(input_features: str, label: str, explanation: str, eos_token: str = "") -> str:
    return format_prompt(input_features) + format_answer(label, explanation) + eos_token
