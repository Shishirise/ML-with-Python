## Definations
```
LangChain is a powerful open source framework designed to help developers build applications using
large language models (LLMs) like GPT, Claude, Mistral, or Gemini with tools, memory, and realworld data.
```
 # Applications
```
Send prompts to LLMs

Combine tools like Python, Google Search, etc.

Add memory to chatbots

Build smart agents that take actions
```
# What exactly is LLMChain? How is it built, how is it connected to models, and how does it actually work behind the scenes?
```
 1. What Is LLMChain?
LLMChain is a LangChain class that connects:

A PromptTemplate → the text the model will see

A LLM (like GPT-3.5) → the model that responds

Optional memory → to remember things (chatbot style)

It acts like a pipeline:

inputs → [PromptTemplate] → [LLM] → output

```
