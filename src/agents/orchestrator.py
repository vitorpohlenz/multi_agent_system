from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Literal

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# Ignore LangChainDeprecationWarning, LLMChain will be deprecated in the future. New approach is to use LangGraph.
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

Intent = Literal["hr", "tech", "finance", "unknown"]


INTENT_SYSTEM_PROMPT = """
You are an intent classifier for an internal SaaS company support system.

You must map each user query to exactly ONE of the following categories:

- hr      : HR, people, benefits, PTO, payroll, holidays, performance reviews,
            hiring, onboarding, offboarding, internal mobility, policies about
            working hours, parental leave, etc.
- tech    : IT support, SaaS product issues, login/auth, SSO, VPN, devices,
            laptops, network, bugs, outages, how to use product features, etc.
- finance : Invoices, billing, pricing, refunds, expense reimbursement, purchase
            orders, budget approvals, cost centers, corporate cards, etc.
- unknown : Anything else that clearly does not fit the above.

Return ONLY ONE intent label: hr, tech, finance, unknown.
"""


USER_PROMPT = """
User query:
{query}

What is the single best intent category?
"""


@dataclass
class OrchestratorResult:
    intent: Intent
    answer: str
    raw_intent_output: str


class Orchestrator:
    """
    Orchestrates routing between specialized RAG agents based on intent classification.
    """

    def __init__(
        self,
        hr_agent: Callable[[str], Dict],
        tech_agent: Callable[[str], Dict],
        finance_agent: Callable[[str], Dict],
        llm: ChatOpenAI,
        langfuse_handler: Optional[BaseCallbackHandler] = None
    ):
        """
        Parameters
        ----------
        hr_agent : Callable
            A callable with signature qa({"query": str}) for HR questions.
        tech_agent : Callable
            A callable with signature qa({"query": str}) for Tech/IT questions.
        finance_agent : Callable
            A callable with signature qa({"query": str}) for Finance questions.
        langfuse_handler : BaseCallbackHandler, optional
            Langfuse callback handler for tracing.
        model_name : str, optional
            LLM name for intent classification.
        """
        self.hr_agent = hr_agent
        self.tech_agent = tech_agent
        self.finance_agent = finance_agent

        callbacks = [langfuse_handler] if langfuse_handler else None

        self.intent_llm = llm

        prompt = PromptTemplate(
            template=INTENT_SYSTEM_PROMPT + USER_PROMPT,
            input_variables=["query"],
        )

        self.intent_chain = LLMChain(
            llm=self.intent_llm,
            prompt=prompt,
            callbacks=callbacks,
            name="Orchestrator Intent Classifier",
        )

    def classify_intent(self, query: str) -> (Intent, str):
        """
        Run the intent classifier and normalize the category label.
        """
        raw_output = self.intent_chain.run(query=query)
        label = raw_output.strip().lower()

        if "hr" == label:
            intent: Intent = "hr"
        elif "tech" == label or "it" == label:
            intent = "tech"
        elif "finance" == label or "billing" in label:
            intent = "finance"
        elif label in {"unknown", "other"}:
            intent = "unknown"
        else:
            # Fallback heuristic based on keywords (defensive classification)
            q = query.lower()
            if any(k in q for k in ["benefit", "payroll", "vacation", "holiday", "hire", "manager", "leave"]):
                intent = "hr"
            elif any(k in q for k in ["login", "password", "bug", "error", "vpn", "laptop", "network", "outage"]):
                intent = "tech"
            elif any(k in q for k in ["invoice", "billing", "refund", "expense", "reimbursement", "po", "budget"]):
                intent = "finance"
            else:
                intent = "unknown"

        return intent, raw_output

    def route(self, query: str) -> OrchestratorResult:
        """
        Classify the intent and route to the appropriate RAG agent.

        Returns
        -------
        OrchestratorResult
            Contains intent, answer text, and raw classifier output.
        """
        intent, raw_output = self.classify_intent(query)

        if intent == "hr":
            response = self.hr_agent.invoke({"query": query})
            answer = response["result"]
        elif intent == "tech":
            response = self.tech_agent.invoke({"query": query})
            answer = response["result"]
        elif intent == "finance":
            response = self.finance_agent.invoke({"query": query})
            answer = response["result"]
        else:
            # Unknown: respond safely and suggest categories
            answer = (
                "I could not confidently route your question to HR, IT, or Finance. "
                "Please clarify whether it relates to:\n"
                "- HR (people, benefits, PTO, payroll)\n"
                "- IT / Tech (product issues, login, devices, outages)\n"
                "- Finance (invoices, billing, expenses, budgets)"
            )

        return OrchestratorResult(
            intent=intent,
            answer=answer,
            raw_intent_output=raw_output,
        )
