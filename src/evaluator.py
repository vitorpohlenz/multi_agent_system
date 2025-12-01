"""
Evaluator script for the multi-agent support system.

How to run:
    python -m src.evaluator --test-file test/test_queries.json

How it works:
- Consumes test queries from a JSON file with entries like:
```json

[
  {
    "query": "user question...",
    "expected_intent": "hr/tech/finance/unknown",
    "expected_example_answer": "ideal or sample answer..."
  },
  ...
]
```

For each entry:
1) Uses the orchestration system to generate an answer.
2) Compares the system answer to the expected_example_answer using an LLM.
3) Produces a 0–10 quality score.
4) Logs results and scores into Langfuse (spans/observations + scores).
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from src.multi_agent_system import build_system


EVALUATION_PROMPT = """
You are a strict evaluation assistant.

You will receive:
1) The user's original question.
2) The system's actual answer.
3) An expected example answer that represents a good, policy-aligned response.

Your task:
- Compare the system's answer against the expected example answer.
- Consider these dimensions:
  - Relevance to the question
  - Completeness of the explanation
  - Alignment with the expected answer's meaning and key points
- Ignore minor wording differences as long as meaning is preserved.

Output:
Return a SINGLE integer score between 0 and 10, where:
- 0–3: Bad (misleading, largely incorrect, or misses key points)
- 4–6: Weak (partially correct but missing important details)
- 7–8: Good (mostly correct with minor omissions)
- 9–10: Excellent (highly aligned with the expected answer in content and intent)

ONLY output the integer number. Do NOT include any explanation.

Question:
{question}

System answer:
{system_answer}

Expected example answer:
{expected_answer}
"""


def init_langfuse() -> Optional[Any]:
    """
    Initialize Langfuse v3 client using environment variables.

    Returns
    -------
    client or None
    """
    
    client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), 
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"), 
        host=os.getenv("LANGFUSE_BASE_URL")
    )
    
    return client


def init_evaluator_llm() -> ChatOpenAI:
    """
    Initialize the LLM used for scoring answers.
    """
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0
    )


def parse_score(raw_output: str) -> int:
    """
    Extract an integer score 0–10 from the LLM output.
    """
    raw_output = raw_output.strip()
    digits = "".join(ch for ch in raw_output if ch.isdigit())
    try:
        score = int(digits)
    except ValueError:
        score = 0
    return max(0, min(10, score))


def evaluate_answer(
    llm: ChatOpenAI,
    question: str,
    system_answer: str,
    expected_answer: str,
) -> Tuple[int, str]:
    """
    Use the evaluator LLM to score a single answer.
    """
    prompt = EVALUATION_PROMPT.format(
        question=question,
        system_answer=system_answer,
        expected_answer=expected_answer,
    )
    raw = llm.invoke(prompt).content
    score = parse_score(raw)
    return score, raw


def run_evaluation(test_file: str) -> None:
    """
    Run evaluation over all entries in the given test_queries.json file.
    """
    load_dotenv()

    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if not isinstance(test_cases, list):
        raise ValueError("Test file must contain a JSON array of test cases.")

    # Build orchestrator system (also sets up Langfuse callbacks for LangChain if configured)
    orchestrator = build_system()

    # Our evaluation-level Langfuse client (v3)
    langfuse = init_langfuse()
    evaluator_llm = init_evaluator_llm()

    num_cases = len(test_cases)
    print(f"[evaluator] Loaded {num_cases} test cases from {path}")

    total_score = 0.0
    cases_run = 0

    if langfuse is None:
        # Run without Langfuse spans/scores, still print results
        print("[evaluator] Running WITHOUT Langfuse tracing (client not available).")
        for idx, case in enumerate(test_cases, start=1):
            query = case.get("query", "")
            expected_intent = case.get("expected_intent", "unknown")
            expected_answer = case.get("expected_example_answer", "")

            if not query:
                print(f"[evaluator] Skipping case {idx}: missing 'query'")
                continue

            result = orchestrator.route(query)
            system_intent = result.intent
            system_answer = result.answer

            score, _ = evaluate_answer(
                evaluator_llm, query, system_answer, expected_answer
            )
            total_score += score
            cases_run += 1

            print(f"\nCase {idx}/{num_cases}")
            print(f"Q: {query}")
            print(f"Expected intent:  {expected_intent}")
            print(f"Predicted intent: {system_intent}")
            print(f"Quality score:    {score}/10")
            print("-" * 60)

        avg_score = total_score / cases_run if cases_run else 0.0
        print(
            f"\n[evaluator] Average quality score across {cases_run} cases: {avg_score:.2f}/10"
        )
        return

    # With Langfuse: create a root span for the whole evaluation batch

    with langfuse.start_as_current_observation(
        as_type="span",
        name="evaluation_batch",
        input={"test_file": str(path), "num_cases": num_cases},
    ) as batch_span:
        for idx, case in enumerate(test_cases, start=1):
            query = case.get("query", "")
            expected_intent = case.get("expected_intent", "unknown")
            expected_answer = case.get("expected_example_answer", "")

            if not query:
                print(f"[evaluator] Skipping case {idx}: missing 'query'")
                continue

            # Child span for each evaluation item; shares the same trace as the batch span
            with langfuse.start_as_current_observation(
                as_type="span",
                name="evaluation_item",
                input={
                    "index": idx,
                    "query": query,
                    "expected_intent": expected_intent,
                    "expected_example_answer": expected_answer,
                },
            ) as item_span:
                # Generate system answer
                result = orchestrator.route(query)
                system_intent = result.intent
                system_answer = result.answer

                # Evaluate quality
                score, raw_eval_output = evaluate_answer(
                    evaluator_llm, query, system_answer, expected_answer
                )

                total_score += score
                cases_run += 1

                # Update span with outputs + metadata
                item_span.update(
                    output={
                        "system_intent": system_intent,
                        "system_answer": system_answer,
                        "score": score,
                        "raw_evaluator_output": raw_eval_output,
                    },
                    metadata={
                        "expected_intent": expected_intent,
                        "predicted_intent": system_intent,
                        "intent_correct": system_intent == expected_intent,
                        "score": score,
                    },
                )

                # Attach a NUMERIC score for this observation
                item_span.score(
                    name="answer_quality",
                    value=float(score),
                    data_type="NUMERIC",
                    comment=f"Evaluator LLM output: {raw_eval_output}",
                )

                # Console output
                print(f"\nCase {idx}/{num_cases}")
                print(f"Q: {query}")
                print(f"Expected intent:  {expected_intent}")
                print(f"Predicted intent: {system_intent}")
                print(f"Quality score:    {score}/10")
                print("-" * 60)

        avg_score = total_score / cases_run if cases_run else 0.0

        # Record aggregate result on the batch span / trace
        batch_span.update(
            output={"average_score": avg_score, "num_cases": cases_run},
            metadata={"average_score": avg_score},
        )
        batch_span.score_trace(
            name="batch_average_quality",
            value=float(avg_score),
            data_type="NUMERIC",
            comment="Average per-answer quality score across all evaluated test cases.",
        )

    # Flush in short-lived scripts so data is actually sent
    langfuse.flush()

    print(
        f"\n[evaluator] Average quality score across {cases_run} cases: {avg_score:.2f}/10"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent system answers against expected example answers."
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="test_queries.json",
        help="Path to test_queries.json.",
    )

    args = parser.parse_args()
    run_evaluation(args.test_file)
