1. `uv init --python 3.12`
2. `uv sync`
3. `source .venv/bin/activate` 
4. `uv add ipykernel` 
5. `uv add python-dotenv` 
6. `touch .env`
7. `uv add openai==0.28`
8. `uv add langchain`
9. `uv add langchain-community`
10. `uv add tiktoken`
11. `uv add pandas`
12. `uv add docarray`
13. `uv add langchain-google-genai`
14. `uv add wikipedia`
15. `uv add langchain-experimental`
16. `uv add numexpr`
17. `uv add DateTime`

## Notebook Overview

The `notebooks/` directory contains a short learning path that introduces key LangChain building blocks. Each notebook focuses on one concept and provides runnable examples you can adapt for your own projects.

| Notebook | Concept | Why it matters |
| --- | --- | --- |
| `A1-Model_prompt_parser.ipynb` | Models, prompts, and output parsers | Demonstrates how to call language models directly or through LangChain abstractions, then normalize their responses with output parsers so downstream code can rely on consistent structure. |
| `A2-Memory.ipynb` | Conversation memory | Shows several memory modules (buffer, window, token-aware, and summarized) that help chatbots maintain context over multi-turn conversations without exceeding token limits. |
| `A3-Chains.ipynb` | Chains | Walks through composing prompts and models into reusable chains, enabling you to sequence tasks and build higher-level workflows from simple components. |
| `A4-QnA.ipynb` | Question answering over documents | Explains how to load documents, embed them, and perform retrieval-augmented generation so users can ask natural-language questions about your data. |
| `A5-Evaluation.ipynb` | Evaluation | Covers generating test examples and using manual or LLM-assisted evaluators to measure and improve the quality of your LangChain applications. |
