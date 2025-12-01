"""
Entry point for the multi-agent routing system.

Usage examples:

    # Single query
    python -m src.multi_agent_system --query "How do I submit an expense report?"

    # Interactive mode
    python -m src.multi_agent_system --interactive

    # Evaluate sample test queries
    python -m src.multi_agent_system --run-test-queries test_queries.json
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))

from agents.agents_components import default_llm
from agents.hr_agent import build_hr_agent
from agents.tech_agent import build_tech_agent
from agents.finance_agent import build_finance_agent
from agents.orchestrator import Orchestrator


def init_langfuse_handler():
    """
    Initialize Langfuse and return a callback handler if credentials are present.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL")

    if not (public_key and secret_key and host):
        print("[multi_agent_system] Langfuse not configured - tracing disabled.")
        return None

    _ = Langfuse(  # noqa: F841
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    handler = LangfuseCallbackHandler()
    print("[multi_agent_system] Langfuse tracing enabled.")
    return handler


def build_system():
    """
    Build all agents and the orchestrator.
    """
    langfuse_handler = init_langfuse_handler()
    hr_qa = build_hr_agent(langfuse_handler=langfuse_handler)
    tech_qa = build_tech_agent(langfuse_handler=langfuse_handler)
    finance_qa = build_finance_agent(langfuse_handler=langfuse_handler)


    orchestrator = Orchestrator(
        hr_agent=hr_qa,
        tech_agent=tech_qa,
        finance_agent=finance_qa,
        llm=default_llm,
        langfuse_handler=langfuse_handler,
    )
    return orchestrator


def run_single_query(orchestrator: Orchestrator, query: str):
    result = orchestrator.route(query)
    print(f"[INTENT]: {result.intent}")
    print("-" * 60)
    print(result.answer)
    print("-" * 60)


def run_interactive(orchestrator: Orchestrator):
    print("Multi-Agent Support System (HR / Tech / Finance)")
    print("Type 'exit()' or 'quit()' to stop.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() in {"exit()", "quit()"}:
            print("Bye!")
            break

        result = orchestrator.route(query)
        print(f"[INTENT]: {result.intent}")
        print("Assistant:", result.answer)


def run_test_queries(orchestrator: Orchestrator, path: str):
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        queries = json.load(f)

    total = len(queries)
    correct = 0

    print(f"Running {total} test queries from {path_obj}...\n")

    for item in queries:
        query = item["query"]
        expected = item["expected_intent"]

        result = orchestrator.route(query)
        predicted = result.intent
        is_correct = predicted == expected
        correct += int(is_correct)

        print(f"Q: {query}")
        print(f"   expected: {expected} | predicted: {predicted} | ok={is_correct}")
        print("-" * 60)

    accuracy = correct / total if total else 0
    print(f"\nIntent classification accuracy: {accuracy:.2%} ({correct}/{total})")


def main():
    load_dotenv()
    print("[multi_agent_system] initializing...")

    parser = argparse.ArgumentParser(description="Multi-agent routing system")
    parser.add_argument("--query", type=str, help="Single query to route")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument(
        "--run-test-queries",
        type=str,
        help="Path to test_queries.json to evaluate intent routing",
    )
    print("[multi_agent_system] args parsed")

    args = parser.parse_args()

    print("[multi_agent_system] building system...")
    orchestrator = build_system()

    if args.query:
        run_single_query(orchestrator, args.query)
    elif args.interactive:
        run_interactive(orchestrator)
    elif args.run_test_queries:
        run_test_queries(orchestrator, args.run_test_queries)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
