# Topic 5 — Generative AI in Code

## Generation as an action, with validation and grounding

This project implements the Topic 5 assessment pipeline: structured generation, bounded repair, retrieval-grounded generation with citation validation, hallucination measurement, and code generation with automatic unit-test execution.

## Project structure

- `schema_gen.py` — Pydantic ReleaseNote schema and 20 schema-valid generations.
- `repair.py` — generate → validate → repair loop, capped at 3 attempts.
- `grounding.py` — TF-IDF retrieval, grounded release notes, citation-ID validation and failure demonstration.
- `hallucination.py` — 30-note / 90-claim held-out factuality evaluation and mitigation comparison.
- `codegen.py` — code generation, AST safety checks, disposable sandbox execution, and bounded repair.
- `outputs/` — generated result summaries and evaluation artifacts.
- `traces/` — structured execution traces.
- `docs/` — submission brief and reflection.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and add your OpenAI API key. Never commit `.env`.

## Run each task

```bash
python schema_gen.py
python repair.py
python grounding.py
python hallucination.py
python codegen.py
```

## Evidence

Task 1 produced 20/20 schema-valid notes and a deliberately invalid raw output that Pydantic rejected. Task 2 generated 30 notes with a maximum of three attempts and deterministic validation-failure injection to exercise repair. Task 3 uses a 24-document corpus, retrieves top-k evidence, validates citation IDs, and demonstrates rejection of an invented citation. Task 4 evaluates 30 notes and exactly 90 claims. The observed hallucination rate was 100.00% in the controlled baseline and 8.89% after mitigation, a reduction of 91.11 percentage points. The baseline intentionally contains one controlled unsupported claim per note and is not presented as naturally occurring model hallucination. Task 5(c) generated implementations plus pytest tests for three specifications; the final run passed 3/3 samples.

## Safety

Generated code is parsed with Python AST checks before execution and is run only in a disposable temporary directory with a timeout. Dangerous imports/calls are rejected and globally installed pytest plugins are disabled. This is a disposable execution environment, not a hardened security boundary; production systems should use container/VM isolation with stronger OS-level controls.

## Reflection

The main lesson from this project is that generation becomes more reliable when it is treated as an action inside a validation loop rather than as a one-shot text completion. Pydantic provided a machine-checkable contract for structured release notes, while the repair task demonstrated how validation errors can become actionable feedback for a bounded regeneration loop. Grounding added a second layer of control: retrieval supplied source evidence and citation IDs were validated rather than merely trusting citations returned by the model. The hallucination experiment made the effect measurable at claim level. Under the controlled baseline, the evaluation deliberately inserted one unsupported feature claim into each note so that mitigation could be reproduced consistently; with lower temperature, strict source-only generation, mandatory citations, and a factuality critic, the measured hallucination rate fell from 100.00% to 8.89%. Task 5(c) extended the same principle to executable artifacts. The model generated both implementation and tests, AST checks rejected unsafe constructs, and the tests ran in a disposable temporary sandbox with a bounded timeout. The final three examples all passed. A key limitation is that the sandbox is disposable rather than a hardened container, so it should not be treated as sufficient isolation for arbitrary hostile code. Overall, the project demonstrates a practical pattern for safer generative systems: generate, validate, ground, execute only after checks, observe the result, and regenerate only within explicit limits.
