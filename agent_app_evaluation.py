from __future__ import annotations

import contextvars
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# 1. Trace context
# We use ContextVar so every evaluation case gets its own trace.
# Tools can write to the currently active trace without receiving it
# explicitly as a function argument.

CURRENT_TRACE: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "CURRENT_TRACE",
    default=None,
)

def now_ms() -> int:
    return int(time.time() * 1000)

def record_event(event_type: str, payload: Dict[str, Any]) -> None:
    trace = CURRENT_TRACE.get()
    if trace is None:
        return

    trace["events"].append(
        {
            "time_ms": now_ms(),
            "type": event_type,
            "payload": payload,
        }
    )

# 2. Small course policy corpus
COURSE_DOCS = [
    {
        "doc_id": "DOC_REFUND",
        "title": "Refund Policy",
        "text": (
            "Students are eligible for a 100% refund if they cancel at least "
            "7 calendar days before the cohort start date. Students are eligible "
            "for a 50% refund within the first 3 calendar days after the cohort "
            "starts. No refund is available after the third calendar day."
        ),
    },
    {
        "doc_id": "DOC_PAUSE",
        "title": "Course Pause Policy",
        "text": (
            "Learners can pause their course once for up to 14 calendar days. "
            "The pause request must be approved by the academic coordinator."
        ),
    },
    {
        "doc_id": "DOC_PROJECTS",
        "title": "Project Submission Policy",
        "text": (
            "All projects must be submitted through the learning portal. "
            "Submissions through email or chat are not considered official."
        ),
    },
    {
        "doc_id": "DOC_PLACEMENT",
        "title": "Career Support Policy",
        "text": (
            "The program provides career support, mock interviews, resume reviews, "
            "and referral guidance. The program does not guarantee a job, salary, "
            "promotion, or interview shortlist."
        ),
    },
]

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def simple_keyword_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Beginner-friendly retrieval implementation.

    In production, replace this with a vector DB like Chroma db.
    """
    query_terms = tokenize(query)
    scored_docs = []

    for doc in COURSE_DOCS:
        doc_terms = tokenize(doc["title"] + " " + doc["text"])
        score = len(query_terms.intersection(doc_terms))

        scored_docs.append(
            {
                **doc,
                "score": score,
            }
        )

    scored_docs.sort(key=lambda d: d["score"], reverse=True)
    return [doc for doc in scored_docs[:k] if doc["score"] > 0]

# 3. Tools
@tool
def search_course_policy(query: str) -> str:
    """
    Search the course policy knowledge base.

    Use this tool for questions about refund, pause, project submission,
    career support, placement guarantees, or other course policy details.
    """
    start = now_ms() # record the current number of milliseconds

    record_event(
        "tool_start",
        {
            "name": "search_course_policy",
            "input": {"query": query},
        },
    )

    docs = simple_keyword_search(query, k=3)

    trace = CURRENT_TRACE.get()
    if trace is not None:
        trace["tool_calls"].append(
            {
                "name": "search_course_policy",
                "input": {"query": query},
                "output_preview": [doc["doc_id"] for doc in docs],
                "latency_ms": now_ms() - start,
            }
        )

        trace["retrievals"].append(
            {
                "query": query,
                "documents": [
                    {
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "score": doc["score"],
                        "text": doc["text"],
                    }
                    for doc in docs
                ],
            }
        )

    if not docs:
        result = "No relevant policy documents found."
    else:
        result = "\n\n".join(
            f"[{doc['doc_id']}] {doc['title']}: {doc['text']}"
            for doc in docs
        )

    record_event(
        "tool_end",
        {
            "name": "search_course_policy",
            "output": result[:500],
            "latency_ms": now_ms() - start,
        },
    )

    return result

@tool
def calculate_refund_amount(fee_amount: float, refund_percent: float) -> str:
    """
    Calculate refund amount from fee amount and refund percentage.

    Use this tool when the user asks for a numerical refund amount.
    """
    start = now_ms()

    record_event(
        "tool_start",
        {
            "name": "calculate_refund_amount",
            "input": {
                "fee_amount": fee_amount,
                "refund_percent": refund_percent,
            },
        },
    )

    refund = fee_amount * refund_percent / 100

    result = (
        f"Refund amount = {fee_amount} * {refund_percent}% = {refund:.2f}"
    )

    trace = CURRENT_TRACE.get()
    if trace is not None:
        trace["tool_calls"].append(
            {
                "name": "calculate_refund_amount",
                "input": {
                    "fee_amount": fee_amount,
                    "refund_percent": refund_percent,
                },
                "output_preview": result,
                "latency_ms": now_ms() - start,
            }
        )

    record_event(
        "tool_end",
        {
            "name": "calculate_refund_amount",
            "output": result,
            "latency_ms": now_ms() - start,
        },
    )

    return result

# 4. Agent
SYSTEM_PROMPT = """
You are a course support assistant.

Rules:
1. Use search_course_policy for course policy questions.
2. Use calculate_refund_amount for numerical refund calculations.
3. If the answer is based on a policy document, cite the document id, for example [DOC_REFUND].
4. If the policy is not found in the retrieved documents, say you do not have enough information.
5. Do not invent personal information, phone numbers, private data, guarantees, or unsupported policy rules.
6. Keep answers concise and helpful.
"""

def build_agent():
    model_name = "gpt-5.5"

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[search_course_policy, calculate_refund_amount],
        system_prompt=SYSTEM_PROMPT,
    )


def extract_final_text(agent_result: Dict[str, Any]) -> str:
    """
    LangChain agents usually return a state dict containing messages.
    The final message is generally the last message.
    """
    messages = agent_result.get("messages", [])
    if not messages:
        return str(agent_result)

    last_message = messages[-1]
    content = getattr(last_message, "content", None)

    if content is None:
        return str(last_message)

    if isinstance(content, str):
        return content

    return str(content)


def run_agent_case(agent, case_id: str, user_input: str) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "case_id": case_id,
        "input": user_input,
        "started_at_ms": now_ms(),
        "events": [],
        "tool_calls": [],
        "retrievals": [],
        "final_response": None,
        "error": None,
        "latency_ms": None,
    }

    token = CURRENT_TRACE.set(trace)

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            }
        )

        final_text = extract_final_text(result)

        trace["final_response"] = final_text
        trace["latency_ms"] = now_ms() - trace["started_at_ms"]

        return {
            "final_response": final_text,
            "trace": trace,
        }

    except Exception as exc:
        trace["error"] = repr(exc)
        trace["latency_ms"] = now_ms() - trace["started_at_ms"]

        return {
            "final_response": "",
            "trace": trace,
        }

    finally:
        CURRENT_TRACE.reset(token)

