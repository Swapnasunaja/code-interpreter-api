import os
import sys
import json
import traceback
from io import StringIO
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI()


# Allow the assignment website to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request / response models
# ---------------------------------------------------------

class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------------------------------------------------------
# Execute submitted Python code
# ---------------------------------------------------------

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    captured_stdout = StringIO()
    captured_stderr = StringIO()

    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    try:
        exec(code, {})

        output = captured_stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        error_output = traceback.format_exc()

        return {
            "success": False,
            "output": error_output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ---------------------------------------------------------
# Ask AI Pipe which submitted code line caused the error
# ---------------------------------------------------------

def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError(
            "AIPIPE_TOKEN environment variable is not configured."
        )

    prompt = f"""
You are analyzing Python code that failed during execution.

Identify ONLY the line number or line numbers in the USER'S SUBMITTED
PYTHON CODE that caused the runtime error.

Return ONLY valid JSON.

The JSON must have exactly this structure:

{{"error_lines": [3]}}

Rules:
- Use line numbers from the submitted code.
- Line numbering starts at 1.
- Return only the line or lines responsible for the error.
- Do not include explanations.
- Do not include markdown.
- Do not include ```json fences.
- Return valid JSON only.

SUBMITTED CODE:
----------------
{code}
----------------

TRACEBACK:
----------------
{traceback_text}
----------------
"""

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You identify the exact source-code line numbers "
                        "responsible for Python runtime errors. "
                        "Always return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        },
        timeout=30
    )

    # Helpful debugging if AI Pipe rejects the request
    if not response.ok:
        print(
            "AI Pipe error:",
            response.status_code,
            response.text
        )

    response.raise_for_status()

    data = response.json()

    content = data["choices"][0]["message"]["content"].strip()

    # In case the LLM still adds markdown fences
    if content.startswith("```json"):
        content = content[len("```json"):]

    if content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:
        parsed_json = json.loads(content)

        analysis = ErrorAnalysis(**parsed_json)

        return analysis.error_lines

    except Exception as e:
        print("LLM returned:", content)
        raise RuntimeError(
            f"Could not parse AI response: {e}"
        )


# ---------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # IMPORTANT:
    # If code succeeds, DO NOT call the AI.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Only call AI when Python produced an error
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )

    except Exception as e:
        print("AI analysis failed:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"AI error analysis failed: {str(e)}"
        )

    return {
        "error": error_lines,
        "result": execution["output"]
    }


# ---------------------------------------------------------
# Optional health check
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Code interpreter API is running."
    }