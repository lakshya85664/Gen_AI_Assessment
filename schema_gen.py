from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in your local .env file.")

client = OpenAI(api_key=API_KEY)

OUTPUT_DIR = Path("outputs")
TRACE_DIR = Path("traces")
OUTPUT_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)


class ReleaseNote(BaseModel):
    title: str = Field(max_length=80)
    audience: Literal["developer", "end-user", "admin"]
    summary: str = Field(min_length=40, max_length=400)
    changes: list[str] = Field(min_length=2, max_length=8)
    breaking: bool
    citations: list[str]


def generate_release_note(context: str) -> tuple[str, ReleaseNote]:
    schema = ReleaseNote.model_json_schema()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a product release note. Return ONLY JSON "
                    "matching the supplied schema. Do not add markdown "
                    "or commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Product context:\n{context}\n\n"
                    f"Required schema:\n{json.dumps(schema)}"
                ),
            },
        ],
    )

    raw = response.choices[0].message.content or ""
    validated = ReleaseNote.model_validate_json(raw)
    return raw, validated


def rejected_output_demo() -> dict:
    invalid_raw = json.dumps({
        "title": "Bad release note",
        "audience": "customer",
        "summary": "Too short.",
        "changes": ["Only one change"],
        "breaking": False,
        "citations": [],
    })

    try:
        ReleaseNote.model_validate_json(invalid_raw)
    except ValidationError as exc:
        rejection = {
            "status": "REJECTED",
            "raw_output": invalid_raw,
            "validation_errors": exc.errors(),
        }
        (TRACE_DIR / "schema_rejection.json").write_text(
            json.dumps(rejection, indent=2), encoding="utf-8"
        )
        return rejection

    raise AssertionError("Invalid fixture was unexpectedly accepted.")


def main() -> None:
    print("=== TOPIC 5 / TASK 1: SCHEMA-VALID GENERATION ===")
    print(f"Model: {MODEL}")

    rejection = rejected_output_demo()
    print("\n--- Rejection demonstration ---")
    print("Status:", rejection["status"])
    print("Invalid raw output was rejected by Pydantic.")

    contexts = [
        "The dashboard now supports exporting filtered reports as CSV.",
        "The API adds pagination to the activity endpoint.",
        "Administrators can configure session timeout from settings.",
        "The mobile app adds offline viewing for saved articles.",
        "The search page now supports quoted exact-match searches.",
        "The billing page displays downloadable invoices.",
        "The notification system adds configurable email preferences.",
        "The editor now restores the last saved draft automatically.",
        "The API adds a health endpoint for service monitoring.",
        "Users can now archive old projects from the project menu.",
        "The admin console adds a user activity audit view.",
        "The SDK adds a helper for uploading multiple files.",
        "The reports page adds date-range filtering.",
        "The application adds keyboard shortcuts for common actions.",
        "The team page adds role descriptions beside permissions.",
        "The login flow adds clearer password reset guidance.",
        "The settings page adds a language selection control.",
        "The API documentation adds examples for authentication errors.",
        "The workspace adds a duplicate-project action.",
        "The notification center now groups related alerts.",
    ]

    generated = []

    for index, context in enumerate(contexts, start=1):
        try:
            raw, note = generate_release_note(context)
            generated.append(note.model_dump())
            print(f"[{index:02d}/20] VALID | title={note.title!r} | audience={note.audience}")
        except (ValidationError, json.JSONDecodeError) as exc:
            print(f"[{index:02d}/20] REJECTED | {exc}")

    if len(generated) != 20:
        raise RuntimeError(f"Expected 20 valid notes, got {len(generated)}.")

    output = {
        "model": MODEL,
        "count": 20,
        "schema": ReleaseNote.model_json_schema(),
        "notes": generated,
    }

    (OUTPUT_DIR / "generated_notes.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    trace = {
        "task": "schema_valid_generation",
        "model": MODEL,
        "generated_count": 20,
        "all_outputs_schema_valid": True,
        "rejection_demo": rejection,
    }

    (TRACE_DIR / "schema_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )

    print("\n=== FINAL RESULT ===")
    print("Valid generated notes: 20")
    print("All accepted notes are Pydantic-valid.")
    print("Saved: outputs/generated_notes.json")
    print("Saved: traces/schema_trace.json")
    print("Saved: traces/schema_rejection.json")


if __name__ == "__main__":
    main()