import os
import ast
import sys
import shutil
import tempfile
import subprocess
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 30


# ============================================================
# SAFETY VALIDATION
# ============================================================

BLOCKED_IMPORTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "pathlib",
    "shutil",
    "ctypes",
    "pickle",
    "sys",
}

BLOCKED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
}

BLOCKED_ATTRIBUTES = {
    "__globals__",
    "__builtins__",
    "__subclasses__",
    "__class__",
}


def validate_generated_code(code: str) -> tuple[bool, str]:
    """
    Validate generated Python code before execution.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    for node in ast.walk(tree):

        # ----------------------------------------------------
        # Imports
        # ----------------------------------------------------
        if isinstance(node, ast.Import):

            for alias in node.names:
                root = alias.name.split(".")[0]

                if root in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {root}"

        elif isinstance(node, ast.ImportFrom):

            if node.module:
                root = node.module.split(".")[0]

                if root in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {root}"

        # ----------------------------------------------------
        # Function calls
        # ----------------------------------------------------
        elif isinstance(node, ast.Call):

            if isinstance(node.func, ast.Name):

                if node.func.id in BLOCKED_CALLS:
                    return False, f"Blocked call: {node.func.id}"

            elif isinstance(node.func, ast.Attribute):

                if node.func.attr in BLOCKED_CALLS:
                    return False, f"Blocked call: {node.func.attr}"

                # Prevent nested pytest execution
                if (
                    node.func.attr == "main"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pytest"
                ):
                    return False, "Do not call pytest.main()"

        # ----------------------------------------------------
        # Dangerous dunder attributes
        # ----------------------------------------------------
        elif isinstance(node, ast.Attribute):

            if node.attr in BLOCKED_ATTRIBUTES:
                return False, f"Blocked attribute: {node.attr}"

    return True, "Safety validation passed"


# ============================================================
# GENERATION
# ============================================================

def clean_code(code: str) -> str:
    """
    Remove accidental Markdown code fences.
    """

    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()

    elif code.startswith("```"):
        code = code[3:].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def generate_function(spec: str) -> str:
    """
    Generate Python implementation and pytest tests.
    """

    system_prompt = """
You are an expert Python developer.

Return ONLY complete valid Python code.

The output MUST contain:
1. The requested function.
2. pytest tests for that function.

STRICT RULES:

- No Markdown.
- No code fences.
- No explanations.
- No prose outside Python code.
- Do not call pytest.main().
- Do not use os.
- Do not use subprocess.
- Do not use socket.
- Do not use requests.
- Do not use urllib.
- Do not use pathlib.
- Do not use shutil.
- Do not use ctypes.
- Do not use pickle.
- Do not use sys.
- Do not use eval().
- Do not use exec().
- Do not use compile().
- Do not use open().
- Do not use __import__().
- Do not access dangerous dunder attributes.

IMPORTANT:
The specification is authoritative.

Tests must exactly match the specification.

Do NOT invent additional requirements.

Do NOT contradict examples explicitly given in the specification.

When the specification gives exact valid/invalid examples,
the implementation and tests MUST use those examples exactly.

For floating point comparisons, use pytest.approx().
"""


    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": spec,
            },
        ],
    )

    return clean_code(
        response.choices[0].message.content or ""
    )


# ============================================================
# REPAIR
# ============================================================

def repair_function(
    original_code: str,
    spec: str,
    failure_output: str,
) -> str:
    """
    Repair generated code using the exact pytest failure.
    """

    repair_prompt = f"""
The generated implementation failed its automated pytest tests.

You MUST repair the implementation while preserving the
original specification exactly.

============================================================
ORIGINAL SPECIFICATION
============================================================

{spec}

============================================================
CURRENT GENERATED CODE
============================================================

{original_code}

============================================================
EXACT PYTEST FAILURE OUTPUT
============================================================

{failure_output}

============================================================
REPAIR RULES
============================================================

1. Return ONLY complete Python code.

2. Include:
   - the corrected function
   - all pytest tests

3. Do NOT use Markdown.

4. Do NOT call pytest.main().

5. Do NOT delete tests.

6. Do NOT weaken tests.

7. Do NOT change expected test results.

8. Do NOT invent new requirements.

9. Do NOT remove requirements from the specification.

10. Treat the ORIGINAL SPECIFICATION as authoritative.

11. If an example is explicitly marked valid, it MUST pass.

12. If an example is explicitly marked invalid, it MUST fail
    according to the specification.

13. Fix the implementation instead of changing the tests.

14. Use pytest.approx() for floating-point comparisons.

15. Do not use dangerous imports or calls.

16. The exact pytest failure above is evidence of what
    currently went wrong. Use it to correct the implementation.
"""


    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are repairing Python code.

Return ONLY complete Python code.

Preserve the specification.

Never weaken or delete tests.

Never change expected outputs.

Fix the implementation.

Do not call pytest.main().
""",
            },
            {
                "role": "user",
                "content": repair_prompt,
            },
        ],
    )

    return clean_code(
        response.choices[0].message.content or ""
    )


# ============================================================
# DISPOSABLE SANDBOX
# ============================================================

def run_generated(code: str) -> tuple[bool, str]:
    """
    Execute generated tests in a disposable temporary directory.
    """

    sandbox = Path(
        tempfile.mkdtemp(
            prefix="codegen_sandbox_"
        )
    )

    try:

        generated_file = sandbox / "generated_test.py"

        generated_file.write_text(
            code,
            encoding="utf-8",
        )

        # Safety validation BEFORE execution.
        safe, safety_message = validate_generated_code(code)

        if not safe:
            return False, (
                f"SAFETY REJECTION: {safety_message}"
            )

        env = os.environ.copy()

        # Prevent globally installed pytest plugins
        # from interfering with execution on Windows.
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(generated_file),
        ]

        process = subprocess.run(
            command,
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )

        output = (
            (process.stdout or "")
            + "\n"
            + (process.stderr or "")
        ).strip()

        if process.returncode == 0:
            return True, output

        return False, output[-4000:]

    except subprocess.TimeoutExpired as exc:

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        return False, (
            f"TIMEOUT: tests exceeded "
            f"{TIMEOUT_SECONDS} seconds.\n"
            f"{stdout}\n{stderr}"
        )

    except Exception as exc:

        return False, (
            f"Execution error: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:

        # Always delete sandbox.
        shutil.rmtree(
            sandbox,
            ignore_errors=True,
        )


# ============================================================
# PROCESS SAMPLE
# ============================================================

def process_sample(
    sample_number: int,
    spec: str,
) -> dict:

    print("\n" + "=" * 70)
    print(f"SAMPLE {sample_number}")
    print("=" * 70)

    print("\nGenerating code...")

    code = generate_function(spec)

    history = []
    repair_count = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(
            f"\n================ ATTEMPT "
            f"{attempt}/{MAX_ATTEMPTS} ================"
        )

        safe, safety_message = validate_generated_code(code)

        print(
            f"\nSafety: "
            f"{'PASS' if safe else 'FAIL'}"
        )

        if not safe:

            output = (
                f"SAFETY REJECTION: "
                f"{safety_message}"
            )

            passed = False

        else:

            print(
                "\nExecuting tests in disposable sandbox..."
            )

            passed, output = run_generated(code)

        history.append(
            {
                "attempt": attempt,
                "passed": passed,
                "output": output,
            }
        )

        if passed:

            print("\nTests: PASS")
            print("\nPytest output:")
            print(output)

            return {
                "sample": sample_number,
                "passed": True,
                "attempts": attempt,
                "repairs": repair_count,
                "history": history,
                "final_code": code,
            }

        print("\nTests: FAIL")
        print("\nPytest output:")
        print(output)

        if attempt < MAX_ATTEMPTS:

            print(
                "\nTest failure detected."
            )

            print(
                "Automatic repair triggered..."
            )

            code = repair_function(
                original_code=code,
                spec=spec,
                failure_output=output,
            )

            repair_count += 1

        else:

            print(
                "\nFAILED AFTER MAXIMUM ATTEMPTS"
            )

    return {
        "sample": sample_number,
        "passed": False,
        "attempts": MAX_ATTEMPTS,
        "repairs": repair_count,
        "history": history,
        "final_code": code,
    }


# ============================================================
# TASK 5(c) — THREE CODE GENERATION SAMPLES
# ============================================================

SAMPLES = [

    # ========================================================
    # SAMPLE 1 — PHONE NORMALISATION
    # ========================================================

    """
Create a function:

    normalise_phone(s) -> str | None

The function converts valid UK and US phone numbers into
E.164 format.

IMPORTANT: The following rules are exact and must not be
changed or inferred differently.

------------------------------------------------------------
UK LOCAL NUMBERS
------------------------------------------------------------

A UK local number is valid ONLY when:

- It contains exactly 11 digits.
- The first digit is 0.
- After removing spaces, the value has exactly 11 digits.

Convert it by replacing the leading 0 with +44.

Examples:

    "07123 456 789" -> "+447123456789"
    "01234 567 890" -> "+441234567890"

This input is INVALID because it contains 12 digits:

    "07123 456 7890" -> None

------------------------------------------------------------
US LOCAL NUMBERS
------------------------------------------------------------

A US local number is valid ONLY when:

- It contains exactly 10 digits.
- All characters are digits.

Convert it by adding +1.

Example:

    "1234567890" -> "+11234567890"

------------------------------------------------------------
INTERNATIONAL UK NUMBERS
------------------------------------------------------------

A UK international number must:

- Start with +44.
- Contain exactly 10 digits after +44.
- Therefore contain exactly 13 characters in total.

Example:

    "+441234567890" -> "+441234567890"

This is invalid because it has too many digits:

    "+4412345678901" -> None

------------------------------------------------------------
INTERNATIONAL US NUMBERS
------------------------------------------------------------

A US international number must:

- Start with +1.
- Contain exactly 10 digits after +1.
- Therefore contain exactly 12 characters in total.

Example:

    "+11234567890" -> "+11234567890"

------------------------------------------------------------
INVALID INPUT
------------------------------------------------------------

Return None for:

- empty strings
- whitespace-only strings
- alphabetic text
- malformed phone numbers
- numbers with incorrect lengths
- unsupported country codes

Remove spaces before validating the number.

Do NOT remove other characters such as +.

------------------------------------------------------------
MANDATORY TEST CASES
------------------------------------------------------------

The generated pytest tests MUST include:

    normalise_phone("07123 456 789")
        == "+447123456789"

    normalise_phone("01234 567 890")
        == "+441234567890"

    normalise_phone("1234567890")
        == "+11234567890"

    normalise_phone("+441234567890")
        == "+441234567890"

    normalise_phone("+11234567890")
        == "+11234567890"

    normalise_phone("07123 456 7890")
        is None

    normalise_phone("+4412345678901")
        is None

    normalise_phone("12345")
        is None

    normalise_phone("")
        is None

    normalise_phone("   ")
        is None

    normalise_phone("invalid number")
        is None

Do not alter these expected results.
""",


    # ========================================================
    # SAMPLE 2 — SLUGIFY
    # ========================================================

    """
Create a function:

    slugify_text(s) -> str

The function converts text into a lowercase URL-style slug.

Rules:

1. Convert all alphabetic characters to lowercase.

2. Apostrophes must be removed completely.

   IMPORTANT:

   An apostrophe does NOT become a space.

   Example:

       "It's a test" -> "its-a-test"

   NOT:

       "it-s-a-test"

3. Replace all OTHER punctuation characters with spaces.

4. Preserve letters and numbers.

5. Strip whitespace from the beginning and end.

6. Collapse consecutive whitespace into one separator.

7. Replace whitespace separators with "-".

8. Do not create leading or trailing hyphens.

9. Empty or whitespace-only input returns "".

10. The apostrophe rule has priority over the general
    punctuation rule.

Mandatory examples:

    slugify_text("Hello World")
        == "hello-world"

    slugify_text("It's a test")
        == "its-a-test"

    slugify_text("AI/ML Guide!")
        == "ai-ml-guide"

    slugify_text("  Hello   World  ")
        == "hello-world"

    slugify_text("Apostrophe's Test")
        == "apostrophes-test"

    slugify_text("Numbers 123")
        == "numbers-123"

Include pytest tests covering:

- normal text
- repeated spaces
- punctuation
- numbers
- apostrophes
- empty input

Do not change the expected outputs.
""",


    # ========================================================
    # SAMPLE 3 — PERCENTAGE
    # ========================================================

    """
Create a function:

    percent_to_decimal(s) -> float

The function converts a percentage string into a decimal.

Rules:

1. Strip leading and trailing whitespace.

2. The input MUST end with "%".

3. Remove the "%" symbol.

4. Strip whitespace from the numeric portion.

5. Convert the numeric portion to float.

6. The numeric value must be between 0 and 100 inclusive.

7. Values below 0 or above 100 must raise ValueError.

8. Invalid numeric input must raise ValueError.

9. Return the numeric value divided by 100.

Examples:

    percent_to_decimal("75%") == 0.75

    percent_to_decimal("50%") == 0.50

    percent_to_decimal("0%") == 0.0

    percent_to_decimal("100%") == 1.0

    percent_to_decimal("12.5%") == 0.125

    percent_to_decimal(" 25 % ") == 0.25

Invalid examples:

    percent_to_decimal("75")
        -> ValueError

    percent_to_decimal("abc%")
        -> ValueError

    percent_to_decimal("-1%")
        -> ValueError

    percent_to_decimal("101%")
        -> ValueError

Include pytest tests for valid, invalid, and out-of-range
values.

Use pytest.approx() for floating-point comparisons.

Do not change the expected outputs.
""",
]


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results: list[dict]):

    outputs_dir = Path("outputs")
    traces_dir = Path("traces")

    outputs_dir.mkdir(
        exist_ok=True
    )

    traces_dir.mkdir(
        exist_ok=True
    )

    summary = []

    for result in results:

        summary.append(
            {
                "sample": result["sample"],
                "passed": result["passed"],
                "attempts": result["attempts"],
                "repairs": result["repairs"],
            }
        )

    results_file = (
        outputs_dir
        / "code_generation_results.json"
    )

    trace_file = (
        traces_dir
        / "code_generation_trace.json"
    )

    results_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    trace_file.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    return results_file, trace_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TASK 5(c) — CODE GENERATION + "
        "AUTOMATIC UNIT-TEST EXECUTION"
    )
    print("=" * 70)

    print(
        f"\nModel: {MODEL}"
    )

    print(
        f"Max attempts per sample: "
        f"{MAX_ATTEMPTS}"
    )

    print(
        f"Execution timeout: "
        f"{TIMEOUT_SECONDS} seconds"
    )

    print(
        "Execution environment: "
        "disposable temporary sandbox"
    )

    print(
        "Safety validation: enabled"
    )

    print(
        "Pytest plugin autoload: disabled"
    )

    results = []

    for sample_number, spec in enumerate(
        SAMPLES,
        start=1,
    ):

        result = process_sample(
            sample_number,
            spec,
        )

        results.append(result)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    total_repairs = sum(
        result["repairs"]
        for result in results
    )

    max_attempts_used = max(
        result["attempts"]
        for result in results
    )

    for result in results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"Sample {result['sample']}: "
            f"{status} | "
            f"Attempts: {result['attempts']} | "
            f"Repairs: {result['repairs']}"
        )

    print(
        f"\nSamples executed: "
        f"{len(results)}"
    )

    print(
        f"Samples passed:   "
        f"{passed_count}/{len(results)}"
    )

    print(
        f"Samples repaired: "
        f"{total_repairs}"
    )

    print(
        f"Maximum attempts: "
        f"{max_attempts_used}"
    )

    results_file, trace_file = save_results(
        results
    )

    if passed_count == len(results):

        print(
            "\nTASK 5(c): PASS"
        )

        print(
            "All generated implementations "
            "passed automated unit-test validation."
        )

    else:

        print(
            "\nTASK 5(c): NEEDS REVIEW"
        )

        print(
            "One or more generated implementations "
            "did not pass within the bounded repair loop."
        )

    print(
        f"\nResults saved to: "
        f"{results_file}"
    )

    print(
        f"Trace saved to:   "
        f"{trace_file}"
    )


if __name__ == "__main__":
    main()