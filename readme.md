# LangChain Project Setup

This project demonstrates key LangChain building blocks. Follow these steps to set up your environment:

1.  Initialize `uv` with Python 3.12:
    ```bash
    uv init --python 3.12
    ```
2.  Synchronize dependencies:
    ```bash
    uv sync
    ```
3.  Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```
4.  Install `ipykernel`:
    ```bash
    uv add ipykernel
    ```
5.  Install `python-dotenv`:
    ```bash
    uv add python-dotenv
    ```
6.  Create a `.env` file:
    ```bash
    touch .env
    ```
7.  Install `openai==0.28`:
    ```bash
    uv add openai==0.28
    ```
8.  Install `langchain`:
    ```bash
    uv add langchain
    ```
9.  Install `langchain-community`:
    ```bash
    uv add langchain-community
    ```
10. Install `tiktoken`:
    ```bash
    uv add tiktoken
    ```
11. Install `pandas`:
    ```bash
    uv add pandas
    ```
12. Install `docarray`:
    ```bash
    uv add docarray
    ```
13. Install `langchain-google-genai`:
    ```bash
    uv add langchain-google-genai
    ```
14. Install `wikipedia`:
    ```bash
    uv add wikipedia
    ```
15. Install `langchain-experimental`:
    ```bash
    uv add langchain-experimental
    ```
16. Install `numexpr`:
    ```bash
    uv add numexpr
    ```
17. Install `DateTime`:
    ```bash
    uv add DateTime
    ```

## Notebook Overview

The `notebooks/` directory contains a short learning path that introduces key LangChain building blocks. Each notebook focuses on one concept and provides runnable examples you can adapt for your own projects.

| Notebook | Concept | Why it matters |
|---|---|---|
| [`A1-Model_prompt_parser.ipynb`](notebooks/A1-Model_prompt_parser.ipynb) | Models, prompts, and output parsers | Demonstrates how to call language models directly or through LangChain abstractions, then normalize their responses with output parsers so downstream code can rely on consistent structure. |
| [`A2-Memory.ipynb`](notebooks/A2-Memory.ipynb) | Conversation memory | Shows several memory modules (buffer, window, token-aware, and summarized) that help chatbots maintain context over multi-turn conversations without exceeding token limits. |
| [`A3-Chains.ipynb`](notebooks/A3-Chains.ipynb) | Chains | Walks through composing prompts and models into reusable chains, enabling you to sequence tasks and build higher-level workflows from simple components. |
| [`A4-QnA.ipynb`](notebooks/A4-QnA.ipynb) | Question answering over documents | Explains how to load documents, embed them, and perform retrieval-augmented generation so users can ask natural-language questions about your data. |
| [`A5-Evaluation.ipynb`](notebooks/A5-Evaluation.ipynb) | Evaluation | Covers generating test examples and using manual or LLM-assisted evaluators to measure and improve the quality of your LangChain applications. |
