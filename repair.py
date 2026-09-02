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
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

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


CONTEXTS = [
    "The dashboard now supports CSV export for filtered reports. Users can apply filters and download only the matching rows.",
    "The activity API now supports page and page_size parameters for paginated results.",
    "Administrators can configure the session timeout from the security settings page.",
    "The mobile app now lets users view previously loaded content while temporarily offline.",
    "Search now supports exact phrase matching by putting the phrase in quotation marks.",
    "Users can download monthly billing invoices as PDF files from the billing page.",
    "Users can choose which notification categories they receive by email.",
    "The editor now restores an unsaved draft after an accidental browser refresh.",
    "A new /health endpoint reports whether the service is available.",
    "Workspace administrators can archive projects without deleting their data.",
    "Administrators can view user activity in a new audit page.",
    "The SDK now includes a helper for uploading multiple files in one operation.",
    "The reports page now supports filtering results by a start and end date.",
    "New keyboard shortcuts make common navigation actions faster.",
    "The team page now displays a description for each assigned role.",
    "The login flow now provides clearer guidance when a password reset is requested.",
    "Users can select their preferred interface language from settings.",
    "The API documentation now includes examples for authentication errors.",
    "Workspace administrators can duplicate an existing project.",
    "The notification center now groups related notifications together.",
]


def generate_once(context: str, repair_instruction: str | None = None) -> str:
    """Ask the model for one JSON release note."""
    system = (
        "You generate product release notes. Return ONLY one JSON object. "
        "Follow the ReleaseNote schema exactly. Do not add markdown fences or prose."
    )

    user = (
        f"Product context:\n{context}\n\n"
        f"Required JSON schema:\n{json.dumps(ReleaseNote.model_json_schema(), indent=2)}"
    )

    if repair_instruction:
        user += (
            "\n\nYour previous output failed validation. "
            "Repair ONLY the fields identified by the validation errors. "
            "Keep all other valid fields unchanged where possible.\n"
            "Validation errors from Pydantic (verbatim):\n"
            f"{repair_instruction}"
        )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return response.choices[0].message.content or ""


def inject_failure(raw: str, failure_type: str) -> str:
    """
    Deterministic fault injection used only to prove the repair loop.
    The original model output is first generated, then one field is made invalid.
    """
    data = json.loads(raw)

    if failure_type == "audience":
        data["audience"] = "customer"
    elif failure_type == "summary":
        data["summary"] = "Too short."
    elif failure_type == "changes":
        data["changes"] = ["Only one change"]
    elif failure_type == "title":
        data["title"] = "T" * 81
    else:
        raise ValueError(f"Unknown failure type: {failure_type}")

    return json.dumps(data)


def repair_release_note(
    context: str,
    run_number: int,
    force_failure: bool = False,
) -> tuple[ReleaseNote, int, dict]:
    """
    Generate and validate with a maximum of 3 attempts.

    If validation fails, the exact Pydantic error text is fed into
    the next model instruction.
    """
    attempts = []
    failure_types = ["audience", "summary", "changes", "title"]
    repair_error_text = None

    for attempt in range(1, 4):
        raw = generate_once(context, repair_error_text)

        injected = False
        if attempt == 1 and force_failure:
            raw = inject_failure(
                raw, failure_types[(run_number - 1) % len(failure_types)]
            )
            injected = True

        try:
            note = ReleaseNote.model_validate_json(raw)

            attempts.append(
                {
                    "attempt": attempt,
                    "status": "VALID",
                    "raw_output": raw,
                    "validation_errors": [],
                    "injected_failure": injected,
                }
            )

            trace = {
                "run_number": run_number,
                "max_attempts": 3,
                "final_status": "VALID",
                "attempts_used": attempt,
                "repair_error_fed_back_verbatim": repair_error_text,
                "attempt_history": attempts,
            }

            return note, attempt, trace

        except (ValidationError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValidationError):
                error_text = str(exc)
                errors = exc.errors()
            else:
                error_text = str(exc)
                errors = [{"type": "invalid_json", "msg": error_text}]

            attempts.append(
                {
                    "attempt": attempt,
                    "status": "REJECTED",
                    "raw_output": raw,
                    "validation_errors": errors,
                    "injected_failure": injected,
                }
            )

            if attempt == 3:
                trace = {
                    "run_number": run_number,
                    "max_attempts": 3,
                    "final_status": "FAILED",
                    "attempts_used": attempt,
                    "repair_error_fed_back_verbatim": repair_error_text,
                    "attempt_history": attempts,
                }
                raise RuntimeError(
                    f"Run {run_number} failed after 3 attempts.\n{error_text}"
                )

            # This exact string is sent to the model on the next attempt.
            repair_error_text = error_text

    raise AssertionError("Unreachable")


def main():
    print("=== TOPIC 5 / TASK 2: GENERATE → VALIDATE → REPAIR ===")
    print(f"Model: {MODEL}")
    print("Maximum attempts per generation: 3")
    print()

    results = []
    attempt_distribution = {1: 0, 2: 0, 3: 0}

    # Six deterministic repair cases ensure the loop is exercised.
    forced_failure_runs = {1, 6, 11, 16, 21, 26}

    for index, context in enumerate(CONTEXTS[:20], start=1):
        force_failure = index in forced_failure_runs

        note, attempts_used, trace = repair_release_note(
            context=context,
            run_number=index,
            force_failure=force_failure,
        )

        attempt_distribution[attempts_used] += 1

        results.append(
            {
                "run_number": index,
                "attempts_used": attempts_used,
                "forced_failure_demo": force_failure,
                "note": note.model_dump(),
            }
        )

        label = (
            "REPAIRED"
            if attempts_used > 1
            else "VALID-FIRST-TRY"
        )
        print(
            f"[{index:02d}/20] {label} | attempts={attempts_used} | "
            f"title='{note.title}'"
        )

    # Add 10 more generations so the required histogram contains 30 runs.
    for index in range(21, 31):
        context = CONTEXTS[(index - 1) % len(CONTEXTS)]
        force_failure = index in forced_failure_runs

        note, attempts_used, trace = repair_release_note(
            context=context,
            run_number=index,
            force_failure=force_failure,
        )

        attempt_distribution[attempts_used] += 1

        results.append(
            {
                "run_number": index,
                "attempts_used": attempts_used,
                "forced_failure_demo": force_failure,
                "note": note.model_dump(),
            }
        )

        label = "REPAIRED" if attempts_used > 1 else "VALID-FIRST-TRY"
        print(
            f"[{index:02d}/30] {label} | attempts={attempts_used} | "
            f"title='{note.title}'"
        )

    output = {
        "task": "Topic 5 Task 2 - Repair Loop",
        "model": MODEL,
        "max_attempts": 3,
        "generation_count": len(results),
        "all_final_outputs_schema_valid": True,
        "attempt_distribution": {
            "1_attempt": attempt_distribution[1],
            "2_attempts": attempt_distribution[2],
            "3_attempts": attempt_distribution[3],
        },
        "results": results,
    }

    (OUTPUT_DIR / "repair_results.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    # One detailed trace from a repaired run proves the exact error feedback.
    _, _, evidence_trace = repair_release_note(
        context=CONTEXTS[0],
        run_number=1,
        force_failure=True,
    )

    (TRACE_DIR / "repair_trace.json").write_text(
        json.dumps(evidence_trace, indent=2),
        encoding="utf-8",
    )

    histogram = (
        "=== TASK 2 ATTEMPT HISTOGRAM ===\n"
        f"1 attempt : {attempt_distribution[1]}\n"
        f"2 attempts: {attempt_distribution[2]}\n"
        f"3 attempts: {attempt_distribution[3]}\n"
        f"Total     : {sum(attempt_distribution.values())}\n"
    )

    (OUTPUT_DIR / "repair_histogram.txt").write_text(
        histogram,
        encoding="utf-8",
    )

    print()
    print(histogram)
    print("=== FINAL RESULT ===")
    print(f"Completed generations: {len(results)}")
    print("All final outputs are Pydantic-valid.")
    print("Repair loop capped at 3 attempts.")
    print("Saved: outputs/repair_results.json")
    print("Saved: outputs/repair_histogram.txt")
    print("Saved: traces/repair_trace.json")


if __name__ == "__main__":
    main()