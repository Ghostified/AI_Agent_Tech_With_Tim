# Research Assistant Agent README

## Problem Statement
This code aims to implement a research assistant agent capable of generating structured research responses using large language models. The agent is designed to answer queries while providing detailed output in a predefined format, including topic, summary, sources, and tools used.

## Objectives
1. **Build a structured response system** that enforces consistent output formatting for research queries using Pydantic models.
2. **Implement model flexibility** by supporting multiple LLM providers (OpenAI, Anthropic) through configurable API endpoints and keys.
3. **Create an extensible agent framework** that allows for future integration of research tools and data sources while maintaining verbose execution tracking.



## Summary of How the Code Works

This project creates a **research assistant agent** that uses large language models (LLMs) to answer questions and return **structured responses**.

### Key Steps:

1. **Environment Setup**
   - Loads API keys from `.env` files using `dotenv`.
   - Configures the LLM (currently set to use OpenAI-compatible models via OpenRouter).

2. **Structured Output**
   - Defines a `ResearchResponse` schema using Pydantic to enforce response structure: topic, summary, sources, and tools used.
   - Uses a `PydanticOutputParser` to convert raw model outputs into this structured format.

3. **Prompt Template**
   - Sets up a system prompt instructing the model to act as a research assistant.
   - Includes formatting instructions so the model knows how to structure its response.

4. **Agent System**
   - Creates a basic **tool-calling agent** using LangChain.
   - Currently no tools are added, but the framework is ready for future extensions.

5. **Execution**
   - Runs a sample query: `"What is the capital of France?"`
   - Prints both the raw response and the parsed structured result.

---

Summary  
**You give it a question → it asks an AI model → gets a structured answer with topic, summary, sources, and tools used.**