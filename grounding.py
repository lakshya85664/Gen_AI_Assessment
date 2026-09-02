from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")

client = OpenAI(api_key=API_KEY)

OUTPUT_DIR = Path("outputs")
TRACE_DIR = Path("traces")
OUTPUT_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)


class GroundedReleaseNote(BaseModel):
    title: str = Field(max_length=80)
    audience: Literal["developer", "end-user", "admin"]
    summary: str = Field(min_length=40, max_length=400)
    changes: list[str] = Field(min_length=2, max_length=8)
    breaking: bool
    citations: list[str] = Field(min_length=1)


CORPUS = [
    {"id": "DOC-01", "text": "The reports page now supports CSV export for the currently applied filters. Exported files contain the filtered rows shown in the report."},
    {"id": "DOC-02", "text": "The activity API now accepts page and page_size parameters. The default page_size is 50 and the maximum page_size is 200."},
    {"id": "DOC-03", "text": "Administrators can configure session timeout values from 15 to 120 minutes in the security settings."},
    {"id": "DOC-04", "text": "Mobile users can download selected project content for offline viewing. Changes made offline are synchronized after the device reconnects."},
    {"id": "DOC-05", "text": "Search now supports exact phrase matching when a query is enclosed in quotation marks."},
    {"id": "DOC-06", "text": "Billing users can download monthly invoices as PDF files from the billing history page."},
    {"id": "DOC-07", "text": "Users can configure which email notification categories they receive from Notification Preferences."},
    {"id": "DOC-08", "text": "The editor automatically restores an unsaved draft after an unexpected browser refresh when a recoverable draft exists."},
    {"id": "DOC-09", "text": "A GET /health endpoint returns HTTP 200 when the service is available. It is intended for service monitoring."},
    {"id": "DOC-10", "text": "Projects can be archived from the project actions menu. Archived projects are removed from the active-project list."},
    {"id": "DOC-11", "text": "Administrators can view user activity in the Audit page, including the actor, action, and timestamp."},
    {"id": "DOC-12", "text": "The SDK now includes a multi-file upload helper that accepts a list of file paths and uploads them as one operation."},
    {"id": "DOC-13", "text": "Reports support filtering by a start date and end date. The selected date range is applied before results are displayed."},
    {"id": "DOC-14", "text": "Keyboard shortcuts are available for opening search, saving edits, and moving between common navigation areas."},
    {"id": "DOC-15", "text": "The team page now displays a short role description for each member when a role description has been configured."},
    {"id": "DOC-16", "text": "The password reset screen now provides clearer guidance about entering the email address associated with the account."},
    {"id": "DOC-17", "text": "Users can select their preferred interface language from the language settings control."},
    {"id": "DOC-18", "text": "API authentication documentation now includes examples for invalid, expired, and missing authentication credentials."},
    {"id": "DOC-19", "text": "Workspace administrators can duplicate an existing project from the project actions menu."},
    {"id": "DOC-20", "text": "The notification center groups related notifications together to reduce repeated entries in the feed."},
    {"id": "DOC-21", "text": "The CSV export uses UTF-8 encoding and includes the column headers displayed by the report."},
    {"id": "DOC-22", "text": "The activity API returns a next_page value when more records are available after the current page."},
    {"id": "DOC-23", "text": "Session timeout changes apply to new sessions after the administrator saves the security setting."},
    {"id": "DOC-24", "text": "Offline project content is read-only until the device reconnects and synchronization is completed."},
]


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    texts = [item["text"] for item in CORPUS]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).ravel()
    ranked = sorted(range(len(CORPUS)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": CORPUS[i]["id"],
            "score": round(float(scores[i]), 4),
            "text": CORPUS[i]["text"],
        }
        for i in ranked
    ]


def extract_citation_ids(text: str) -> list[str]:
    return re.findall(r"\[(DOC-\d{2})\]", text)


def validate_grounding(note: GroundedReleaseNote, retrieved: list[dict]) -> None:
    """Hard-fail if citations are not retrieved IDs or changes lack valid citations."""
    valid_ids = {item["id"] for item in retrieved}

    invalid_top_level = [cid for cid in note.citations if cid not in valid_ids]
    if invalid_top_level:
        raise ValueError(
            f"Invalid citation ID(s): {invalid_top_level}. "
            f"Allowed citation IDs: {sorted(valid_ids)}"
        )

    if not note.changes:
        raise ValueError("Grounding failure: no changes were generated.")

    for index, change in enumerate(note.changes, 1):
        cited = extract_citation_ids(change)
        if not cited:
            raise ValueError(
                f"Grounding failure: change {index} has no citation ID. "
                f"Each change must cite one of: {sorted(valid_ids)}"
            )

        invalid = [cid for cid in cited if cid not in valid_ids]
        if invalid:
            raise ValueError(
                f"Invalid citation ID(s) in change {index}: {invalid}. "
                f"Allowed citation IDs: {sorted(valid_ids)}"
            )

        if not any(cid in note.citations for cid in cited):
            raise ValueError(
                f"Grounding failure: citation(s) {cited} in change {index} "
                f"are missing from the top-level citations list."
            )


def generate_grounded_note(
    request: str,
    retrieved: list[dict],
    max_retries: int = 2,
) -> tuple[GroundedReleaseNote, list[dict]]:
    context = "\n".join(
        f"[{item['id']}] {item['text']}" for item in retrieved
    )
    allowed_ids = ", ".join(item["id"] for item in retrieved)

    system_prompt = """You generate product release notes from retrieved source chunks.

STRICT OUTPUT CONTRACT:
Return ONLY one JSON object with EXACTLY these top-level fields:
{
  "title": "string",
  "audience": "developer" | "end-user" | "admin",
  "summary": "string",
  "changes": ["string", "string"],
  "breaking": false,
  "citations": ["DOC-XX"]
}

IMPORTANT:
- Do NOT return version, release_note, items, description objects, citation_ids, or any other fields.
- changes MUST be an array of plain strings, never objects.
- citations MUST be a top-level array of source IDs.
- Every change string MUST end with at least one citation ID in square brackets, such as [DOC-01].
- Use ONLY source IDs supplied in the retrieved chunks.
- Never invent a citation ID.
- Every factual statement must be supported by the supplied chunks.
- Do not add unsupported details.
- breaking must be false unless the supplied source explicitly states a breaking change.
"""

    feedback = ""
    trace = []

    for attempt in range(1, max_retries + 2):
        user_prompt = f"""Release-note request:
{request}

Retrieved source chunks:
{context}

Allowed citation IDs:
{allowed_ids}

{feedback}

Generate the JSON object now. Follow the exact output contract."""

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content or ""

        try:
            note = GroundedReleaseNote.model_validate_json(raw)
            validate_grounding(note, retrieved)

            trace.append({
                "attempt": attempt,
                "status": "VALID",
                "raw_output": raw,
            })
            return note, trace

        except (ValidationError, ValueError) as exc:
            error_text = str(exc)

            trace.append({
                "attempt": attempt,
                "status": "REJECTED",
                "raw_output": raw,
                "validation_error": error_text,
                "feedback_sent_verbatim": error_text,
            })

            if attempt > max_retries:
                raise RuntimeError(
                    f"Grounded generation failed after {max_retries + 1} attempts: "
                    f"{error_text}"
                )

            # The exact validation error is deliberately fed into the next prompt.
            feedback = (
                "The previous output was rejected. Fix ONLY what caused the "
                "validation failure. The validator returned this exact error:\n"
                f"{error_text}"
            )

    raise AssertionError("Unreachable")


def citation_failure_demo() -> dict:
    retrieved = retrieve("CSV export filtered reports", top_k=3)
    valid_ids = {item["id"] for item in retrieved}

    fake_raw = json.dumps({
        "title": "CSV Export for Filtered Reports",
        "audience": "end-user",
        "summary": "Users can now export the rows currently shown after report filters are applied.",
        "changes": [
            f"CSV export respects the currently applied report filters. [{retrieved[0]['id']}]",
            "The export is available from the reports page. [INVENTED-999]",
        ],
        "breaking": False,
        "citations": [retrieved[0]["id"], "INVENTED-999"],
    })

    try:
        note = GroundedReleaseNote.model_validate_json(fake_raw)
        validate_grounding(note, retrieved)
        raise AssertionError("Invented citation was not rejected.")
    except (ValidationError, ValueError) as exc:
        result = {
            "status": "REJECTED",
            "reason": "Invented citation ID was detected.",
            "invalid_citation": "INVENTED-999",
            "allowed_ids": sorted(valid_ids),
            "error": str(exc),
            "raw_output": fake_raw,
        }
        (TRACE_DIR / "grounding_citation_failure.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result


def main() -> None:
    print("=== TOPIC 5 / TASK 3: GROUNDING + CITATIONS ===")
    print(f"Model: {MODEL}")
    print("Retriever: local TF-IDF cosine similarity")
    print(f"Corpus chunks: {len(CORPUS)}")

    queries = [
        "How can users export filtered reports?",
        "What pagination support was added to the activity API?",
        "How can administrators configure session timeout?",
        "How does offline project viewing work?",
        "What changed in search phrase matching?",
    ]

    retrieval_trace = []
    print("\n--- Retrieval output for 5 queries ---")

    for idx, query in enumerate(queries, 1):
        hits = retrieve(query, top_k=3)
        retrieval_trace.append({"query": query, "top_k": hits})
        print(f"\nQuery {idx}: {query}")
        for rank, hit in enumerate(hits, 1):
            print(
                f"  [{rank}] {hit['id']} | score={hit['score']} | {hit['text']}"
            )

    print("\n--- Citation failure demonstration ---")
    failure = citation_failure_demo()
    print("Status:", failure["status"])
    print("Invalid citation:", failure["invalid_citation"])
    print("Invented citation was rejected before acceptance.")

    requests = [
        "Announce the filtered CSV export feature for report users.",
        "Announce the new activity API pagination support for developers.",
        "Announce configurable session timeout for administrators.",
        "Announce offline project viewing for mobile end-users.",
        "Announce exact phrase matching in search for end-users.",
    ]

    notes = []
    print("\n--- Grounded generation ---")

    for idx, request in enumerate(requests, 1):
        hits = retrieve(request, top_k=3)
        try:
            note, trace = generate_grounded_note(request, hits)
            notes.append({
                "request": request,
                "retrieved_chunks": hits,
                "note": note.model_dump(),
                "validation_trace": trace,
            })
            print(
                f"[{idx}/5] VALID | title='{note.title}' | "
                f"citations={note.citations}"
            )
            for change in note.changes:
                print(f"       - {change}")
        except Exception as exc:
            print(f"[{idx}/5] FAILED | {exc}")

    result = {
        "task": "task_3_grounding",
        "model": MODEL,
        "retriever": "TF-IDF cosine similarity",
        "corpus_size": len(CORPUS),
        "retrieval_examples": retrieval_trace,
        "citation_failure_demo": failure,
        "grounded_notes": notes,
        "all_generated_notes_valid": len(notes) == 5,
    }

    (OUTPUT_DIR / "grounding_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (TRACE_DIR / "grounding_trace.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL RESULT ===")
    print(f"Corpus chunks: {len(CORPUS)}")
    print(f"Retrieval queries tested: {len(queries)}")
    print(f"Grounded notes generated: {len(notes)}/5")
    print("Invented citation rejection: PASS")
    print("Saved: outputs/grounding_results.json")
    print("Saved: traces/grounding_trace.json")
    print("Saved: traces/grounding_citation_failure.json")


if __name__ == "__main__":
    main()