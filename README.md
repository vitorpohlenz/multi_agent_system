# Multi-Agent Support Routing System

This repository implements a **multi-agent AI routing system** simulating a mid-sized SaaS company. 
It automatically classifies incoming user queries (HR, IT/Tech, Finance) and sends them to specialized **RAG agents** grounded in internal company documentation. 
It also includes a full evaluation pipeline using **Langfuse** for observability and automated quality scoring.

---

## Project Description & Context

Misrouted support tickets (e.g., HR questions going to IT, Finance to Legal) cause delays, frustration, and operational overhead. 
This project addresses that by using:

### Orchestrator Agent 
Classifies *intent* (`hr`, `tech`, `finance`, `unknown`) and routes queries to the correct specialized agent.

### Domain-Specific RAG Agents 
Each domain uses a FAISS vector store over curated text documents:
- **HR Agent** → PTO, leave, parental policies, working hours, dress code, etc. 
- **Tech Agent** → authentication, VPN, incidents, runbooks, product usage. 
- **Finance Agent** → billing, invoices, discounts, expenses, reimbursements. 

### Langfuse-Powered Observability 
- Every LLM call (classifier + RAG agents) is traced via callbacks. 
- Evaluations create **batch spans** and **item spans** with numeric scores. 
- Engineers can inspect misclassifications, poor retrievals, and low-quality answers.

### Automated Evaluator 
`src/evaluator.py` consumes `test_queries.json` containing `queries`, `expected_intent`, and `expected_example_answer`, and:
- Generates answers via the orchestrator.
- Uses an evaluator LLM to score answer quality from **0–10**.
- Logs per-answer and batch-average scores into Langfuse.

---

# Repository Structure

```text
multi_agent_system/
│
├── README.md           # This file
├── requirements.txt    # dependancies
├── test_queries.json   # sample/test queries with expected behaviour
├── .env.example        # template of `.env` file with environment variables
├── LICENSE             # MIT license
│
├── src/
│  ├── multi_agent_system.py    # main script to run this project
│  ├── evaluator.py             # evaluator script to evaluate test_queries.json
│  └── agents/
│    ├── hr_agent.py            # RAG HR Agent
│    ├── tech_agent.py          # RAG Tech Agent
│    ├── finance_agent.py       # RAG Finance Agent
│    ├── agents_components.py   # Commom components for all agents and orchestrator, used to avoid code repetition
│    └── orchestrator.py        # Orchestrator agent to classify queries and invoke RAG agents
│
└── data/
  ├── hr_docs/          # HR related documents for RAG
  ├── tech_docs/        # Tech related documents for RAG
  └── finance_docs/     # Finance related documents for RAG
```

---

# Setup Instructions

## Python Version
- System tested with Python **3.9.7**

## Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Configure environment variables
Create a `.env` file in the project root folder, following the `.env.example` file

---

# How to Run

All commands assume you are in the project root and your virtual environment is activated.

## Running a single query

```bash
python -m src.multi_agent_system --query "How do I submit an expense report?"
```

The script will:

- Build all agents and the orchestrator.
- Route the query to the appropriate agent.
- Print the predicted intent, answer, and raw classifier output.
- Create a Langfuse trace (using keys from .env file).

## Interactive chat mode

```bash
python -m src.multi_agent_system --interactive
```

Then type questions such as:

- `How many PTO days do I have per year?`
- `My VPN is not connecting when I work from home.`
- `Can we switch to annual billing and get a discount?`

Exit with `exit()` or `quit()`.

## Run answer-quality evaluation

```bash
python -m src.evaluator --test-file test_queries.json
```

This will:

- Use the orchestrator to answer each query.
- Use the evaluator LLM to score answer quality vs the `expected_example_answer`.
- Check expected_intent and classified_intent
- Create a Langfuse **evaluation batch span** and per-case **evaluation item spans** with numeric scores.

---

# Usage Examples

## HR Example

```bash
python -m src.multi_agent_system --query "How many PTO days do I have per year?"
```

Expected: 
- Intent → `hr` 
- Answer grounded in HR PTO policy from `data/hr_docs/`.

## Tech Example

```bash
python -m src.multi_agent_system --query "I can't connect to the VPN, what should I check?"
```

Expected: 
- Intent → `tech` 
- Answer referencing VPN sections from `data/tech_docs/`.

## Finance Example

```bash
python -m src.multi_agent_system --query "Can we switch our billing from monthly to annual?"
```

Expected: 
- Intent → `finance` 
- Answer grounded in billing and pricing policies under `data/finance_docs/`.


# Technical Decisions

Explanations why LangChain components, routing strategies, and RAG configurations were chosen.

## LangChain Components

**Intent Classification with `LLMChain`**

- The orchestrator uses an `LLMChain` with a focused prompt and a small, fixed label set (`hr`, `tech`, `finance`, `unknown`). 
  - `LLMChain` keeps the classification logic transparent and inspectable. 
  - Prompt engineering for classification is easy to iterate on: we can adjust descriptions, add examples, and re-run tests without changing Python logic. 
  - This is preferable to ad‑hoc manual parsing of raw model output, and lighter than building a fully custom tool-based agent just for intent classification.

**Domain Agents with `RetrievalQA`**

- Each HR/Tech/Finance agent is built as a `RetrievalQA` chain. 
   - `RetrievalQA` encapsulates the common RAG pattern: retrieval → answer synthesis → optional source documents. 
   - It integrates naturally with vector stores and supports callback handlers, which are required for Langfuse observability. 
   - Using `RetrievalQA` avoids re-implementing prompt wiring and retrieval logic, reducing surface for bugs and making the system more maintainable.

**FAISS Vector Store via LangChain**

- The agents use `FAISS.from_documents(...)` from `langchain_community.vectorstores`. 
  - FAISS was chosen for this project to be a self contained, whitout being necessary to maintain a full vector database. 
  - LangChain abstracts the store interface, so migrating to a managed store in the future only requires minor changes.

**Callbacks and Langfuse**

- All LangChain components use a **Langfuse callback handler** when Langfuse is configured. 
  - This gives deep visibility into the entire call graph (classification → retrieval → generation) without scattering logging code in each function. 
  - Engineers can debug misrouted queries by inspecting the exact classifier prompt/response and the documents retrieved by the RAG agents.

## Routing Strategy

**Single Orchestrator with Conditional Routing**

- All incoming queries go through a **single orchestrator** that performs intent classification and then conditionally executes exactly one domain RAG agent. 
  - This matches the real-world requirement: a ticket must ultimately be owned by a single department. 
  - A single decision point makes it easy to audit and tune routing behavior (e.g., change thresholds, refine prompts, or add a Legal intent).

**Fixed Label Set + Fallback Heuristics**

- The classifier is instructed to output exactly one of `hr`, `tech`, `finance`, or `unknown`. 
- If the raw output is noisy, the system applies keyword-based heuristics to derive a final intent. 
  - A fixed label set ensures that downstream logic is simple and robust (no need to handle arbitrary free‑text labels). 
  - Heuristics provide a safety net if the model deviates from the prompt instruction, which is critical in production routing scenarios.

**`unknown` Intent as Safety Valve**

- Any query that does not cleanly map to HR, Tech, or Finance is labeled `unknown`, and the user receives a clarifying response. 
  - It is safer to *not* answer than to confidently hallucinate a policy or make an incorrect financial statement. 
  - This follows a conservative design principle: when in doubt, ask for clarification or route to a human.

**One-Call Routing vs Tool-Calling Agent**

- The design uses a simple classifier + explicit routing instead of a complex tool-calling agent that could decide which tools to use. 
  - The major decision here is about code observability and full control over the orchestrator's behavior.
  - For this use case, the routing space is small and well-defined; a dedicated classifier is easier to reason about, tune, and test. 
  - Latency and cost are more predictable: exactly one classification call plus one RAG call per query.

## RAG Configuration Choices

**Per-Domain Knowledge Bases**

- HR, Tech, and Finance each have their own document directories and vector stores. 
  - Isolating domains reduces cross-contamination: e.g., Tech questions being forwarded to Finance Agent. 
  - It mirrors how organizations structure knowledge—different departments own different policy sets. 
  - It also enables department-specific tuning (e.g., HR might prioritize FAQs, Tech might prioritize runbooks).

**Text Loader and Chunks**

- Using `TextLoader()` and `RecursiveCharacterTextSplitter()` from Langchain improves the code modularity and reduces the maintenace costs of custom functions.

**Retriever Settings (`k=5`)**

- The retriever returns the top 5 similar chunks per query. 
  - 1–2 chunks may under-represent policies that are spread across multiple sections. 
  - Many more than 5 chunks would increase token cost and risk distracting the LLM with loosely related text. 
  - `k=5` is a pragmatic default that can be tuned per domain based on real queries.

**Temperature and Determinism**

- Default chat models use `temperature=0.1`. 
  - Support policies, HR rules, and billing answers must be consistent and repeatable. 
  - Lower randomness helps ensure that auditing and debugging in Langfuse is meaningful: the same question should yield the same answer for reproducible traces.

**Evaluator Design**

- The evaluator compares the system answer to an `expected_example_answer` rather than just checking intent or keyword matches. 
  - It measures semantic quality: correctness, completeness, and adherence to policy. 
  - Scores (0–10) are logged as numeric metrics in Langfuse, enabling dashboards and alerts (e.g., flagging runs where average quality drops).

---

## ⚠️ Known Limitations

1. **Cold-start vector stores** 
  - FAISS indexes are rebuilt on every process start. 
  - This is fine for small/medium docs but not optimal for very large knowledge bases.

2. **LLM-Only Intent Classifier** 
  - No explicit confidence scores; we rely on label constraints and heuristics. 
  - In production, you might add a second-pass classifier or threshold for human escalation.

3. **Missing Additional Departments** 
  - Legal, Sales, and Information Security are not yet implemented but follow the same pattern. 
  - Adding a new department means: new docs, new RAG agent, and one more label in the classifier.

4. **Evaluator Depends on Good Reference Answers** 
  - If `expected_example_answer` in `test_queries.json` is weak or incomplete, the score will be biased. 
  - Creating high-quality references is part of the evaluation effort.

5. **No HTTP API Layer Included** 
  - This repo focuses on orchestration, RAG, and evaluation logic. 
  - For production use, you’d typically wrap this in a FastAPI/Flask/other service with auth and rate limiting.

---

## Final Notes

- To change the default LLM model in the entire project edit `LLM_MODEL="model_name"` in the `.env` file.
- The sample answers in `test_queries.json` were generated using `openai/gpt-oss-20b:free` for RAG and Orchestrator, from [OpenRouter](https://openrouter.ai/)
- Running the `src/evaluator.py` for the `test_queries.json` provided, using `openai/gpt-5.1`, you should get results similar to the following:

```text
Case 1/14
Q: How many vacation days do I have per year?
Expected intent:  hr
Predicted intent: hr
Quality score:    10/10
------------------------------------------------------------

Case 2/14
Q: How do I request paid time off?
Expected intent:  hr
Predicted intent: hr
Quality score:    8/10
------------------------------------------------------------

Case 3/14
Q: What is the parental leave directives when my partner gives birth?
Expected intent:  hr
Predicted intent: hr
Quality score:    9/10
------------------------------------------------------------

Case 4/14
Q: Wich are the working hours?
Expected intent:  hr
Predicted intent: hr
Quality score:    10/10
------------------------------------------------------------

Case 5/14
Q: What is the dress code for visitors coming to the office next week?
Expected intent:  hr
Predicted intent: hr
Quality score:    10/10
------------------------------------------------------------

Case 6/14
Q: What are the categories of the severities of incidents in our SaaS?
Expected intent:  tech
Predicted intent: tech
Quality score:    9/10
------------------------------------------------------------

Case 7/14
Q: Can I lend my credentials to a work colleague?
Expected intent:  tech
Predicted intent: tech
Quality score:    10/10
------------------------------------------------------------

Case 8/14
Q: Can I use MFA with a hardware token?
Expected intent:  tech
Predicted intent: tech
Quality score:    10/10
------------------------------------------------------------

Case 9/14
Q: What are the requirements for my new password?.
Expected intent:  tech
Predicted intent: tech
Quality score:    10/10
------------------------------------------------------------

Case 10/14
Q: Where can I download my last invoice and see the breakdown of charges for our subscription?
Expected intent:  finance
Predicted intent: finance
Quality score:    10/10
------------------------------------------------------------

Case 11/14
Q: When the price will be reviewed?
Expected intent:  finance
Predicted intent: finance
Quality score:    10/10
------------------------------------------------------------

Case 12/14
Q: Can we switch our billing from monthly to annual and will there be a discount?
Expected intent:  finance
Predicted intent: finance
Quality score:    9/10
------------------------------------------------------------

Case 13/14
Q: How can I cut the grass?
Expected intent:  unknown
Predicted intent: unknown
Quality score:    10/10
------------------------------------------------------------

Case 14/14
Q: How can I change the windows glass?
Expected intent:  unknown
Predicted intent: unknown
Quality score:    10/10
------------------------------------------------------------

[evaluator] Average quality score across 14 cases: 9.64/10
```
