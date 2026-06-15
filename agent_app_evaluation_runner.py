from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from agent_app_evaluation import build_agent, run_agent_case

RESULTS_PATH = Path("results.csv")
TRACES_DIR = Path("traces")

# Set = {x, y, z, ......,} => set contains unique values.

REFUSAL_PHRASES = [
    "i do not have enough information",
    "i don't have enough information",
    "i can’t provide",
    "i can't provide",
    "not available",
    "not found",
    "cannot provide",
    "do not have access",
    "don't have access",
]

def load_cases(path: str = "eval_cases.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(text: str) -> str:
    return text.lower().strip()

def contains_refusal(text: str) -> bool:
    text = normalize(text)
    return any(phrase in text for phrase in REFUSAL_PHRASES)

def get_tools_used(trace: Dict[str, Any]) -> List[str]:
    return [call["name"] for call in trace.get("tool_calls", [])]

def get_retrieved_doc_ids(trace: Dict[str, Any]) -> List[str]:
    doc_ids = []

    for retrieval in trace.get("retrievals", []):
        for doc in retrieval.get("documents", []):
            doc_ids.append(doc["doc_id"])

    return sorted(set(doc_ids))

def evaluate_case(
    case: Dict[str, Any],
    final_response: str,
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    expected = case["expected"]

    tools_used = get_tools_used(trace)
    retrieved_doc_ids = get_retrieved_doc_ids(trace)
    final_lower = normalize(final_response)

    failures = []

    # 1. Required tools
    for tool_name in expected.get("must_use_tools", []):
        if tool_name not in tools_used:
            failures.append(
                {
                    "type": "TOOL_MISSING",
                    "message": f"Expected tool not used: {tool_name}",
                }
            )

    # 2. Forbidden tools
    for tool_name in expected.get("forbidden_tools", []):
        if tool_name in tools_used:
            failures.append(
                {
                    "type": "FORBIDDEN_TOOL_USED",
                    "message": f"Forbidden tool was used: {tool_name}",
                }
            )

    # 3. Required citations / document ids
    for doc_id in expected.get("must_cite_doc_ids", []):
        if doc_id not in final_response:
            failures.append(
                {
                    "type": "MISSING_CITATION",
                    "message": f"Expected citation missing: {doc_id}",
                }
            )

    # 4. Required answer content
    for phrase in expected.get("must_contain", []):
        if normalize(phrase) not in final_lower:
            failures.append(
                {
                    "type": "CONTENT_MISSING",
                    "message": f"Expected phrase missing: {phrase}",
                }
            )

    # 5. Refusal behavior
    expected_refusal = expected.get("should_refuse", False)
    actual_refusal = contains_refusal(final_response)

    if expected_refusal and not actual_refusal:
        failures.append(
            {
                "type": "REFUSAL_MISSING",
                "message": "Expected refusal, but agent answered.",
            }
        )

    if not expected_refusal and actual_refusal:
        failures.append(
            {
                "type": "OVER_REFUSAL",
                "message": "Agent refused even though it should answer.",
            }
        )

    # 6. Runtime error
    if trace.get("error"):
        failures.append(
            {
                "type": "RUNTIME_ERROR",
                "message": trace["error"],
            }
        )

    status = "PASS" if not failures else "FAIL"
    failure_type = classify_failure_type(failures)
    # if no failure, then score = 1.0
    # if 1 failure, then score = 0.75
    # if 2 failures, then score = 0.5
    # if 10 failures, then score = 0
    score = max(0.0, 1.0 - 0.25 * len(failures))

    return {
        "case_id": case["id"],
        "status": status,
        "score": score,
        "failure_type": failure_type,
        "failures": failures,
        "tools_used": tools_used,
        "retrieved_doc_ids": retrieved_doc_ids,
        "latency_ms": trace.get("latency_ms"),
        "final_response": final_response,
    }

def classify_failure_type(failures: List[Dict[str, str]]) -> str:
    if not failures:
        return "NONE"

    priority = [
        "RUNTIME_ERROR",
        "TOOL_MISSING",
        "FORBIDDEN_TOOL_USED",
        "MISSING_CITATION",
        "REFUSAL_MISSING",
        "OVER_REFUSAL",
        "CONTENT_MISSING",
    ]

    failure_types = {failure["type"] for failure in failures}

    for failure_type in priority:
        if failure_type in failure_types:
            return failure_type

    return "UNKNOWN_FAILURE"

def write_trace(case_id: str, trace: Dict[str, Any]) -> str:
    TRACES_DIR.mkdir(exist_ok=True)

    path = TRACES_DIR / f"{case_id}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return str(path)

def write_results_csv(rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "status",
        "score",
        "failure_type",
        "latency_ms",
        "tools_used",
        "retrieved_doc_ids",
        "final_response",
        "trace_file",
        "failures",
    ]

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "status": row["status"],
                    "score": row["score"],
                    "failure_type": row["failure_type"],
                    "latency_ms": row["latency_ms"],
                    "tools_used": "|".join(row["tools_used"]),
                    "retrieved_doc_ids": "|".join(row["retrieved_doc_ids"]),
                    "final_response": row["final_response"],
                    "trace_file": row["trace_file"],
                    "failures": json.dumps(row["failures"], ensure_ascii=False),
                }
            )

def print_summary(rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row["status"] == "PASS")
    failed = total - passed

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total cases : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if failed:
        print("\nLowest-performing cases:")
        failed_rows = sorted(
            [row for row in rows if row["status"] == "FAIL"],
            key=lambda r: r["score"],
        )

        for row in failed_rows:
            print(
                f"- {row['case_id']} | "
                f"score={row['score']} | "
                f"failure={row['failure_type']} | "
                f"trace={row['trace_file']}"
            )

def main() -> None:
    cases = load_cases()
    agent = build_agent()

    rows = []

    for case in cases:
        print(f"Running case: {case['id']}")

        output = run_agent_case(
            agent=agent,
            case_id=case["id"],
            user_input=case["input"],
        )

        final_response = output["final_response"]
        trace = output["trace"]

        trace_file = write_trace(case["id"], trace)

        evaluation = evaluate_case(
            case=case,
            final_response=final_response,
            trace=trace,
        )

        evaluation["trace_file"] = trace_file
        rows.append(evaluation)

    write_results_csv(rows)
    print_summary(rows)

    print(f"\nSaved results to: {RESULTS_PATH}")
    print(f"Saved traces to : {TRACES_DIR}/")

if __name__ == "__main__":
    main()

# Evaluation Philosophy
# 1. Happy Cases.
# 2. Grounding Cases -> source citing
# 3. Refusal Cases
# 4. Unsupported policy query
# 5. Missing tool

