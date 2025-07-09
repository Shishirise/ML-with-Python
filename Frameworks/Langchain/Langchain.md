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

# 1. What Is LLMChain?
```
LLMChain is a LangChain class that connects:

A PromptTemplate → the text the model will see

A LLM (like GPT-3.5) → the model that responds

Optional memory → to remember things (chatbot style)

It acts like a pipeline:

inputs → [PromptTemplate] → [LLM] → output
```



# 2.How LLMChain Works (Internally)

```python
Define templates

PromptTemplate(input_variables=["name"], template="Tell me about {name}")
```


You connect it to an LLM (GPT,Gemini, Claude, etc.):

```python

llm = OpenAI()
```
You pass both into an LLMChain:

```python

chain = LLMChain(prompt=prompt, llm=llm)
```
You run it like a function:

```python

output = chain.run(name="Einstein")
```
Internally:

Fills the prompt: "Tell me about Einstein"

Sends it to the model using .invoke() or .generate()

Parses and returns the response
```
```
## MEMORY
```
Memory lets an LLM-based app remember past messages, steps, or data across interactions.

LLM can recall what the user said earlier
Enables chatbots, conversational agents, and long interactions
```

# Types of Memory in LangChain

| Memory Type                  | What It Stores                     |
|-----------------------------|----------------------------------|
| ConversationBufferMemory     | Full message history (chat logs) |
| ConversationSummaryMemory    | GPT-generated summary of history |
| ConversationBufferWindowMemory | Only the last N messages         |
| VectorStoreRetrieverMemory   | Stores past data in a vector DB  |
| ReadOnlySharedMemory         | Read-only view of another memory |


```python
from langchain.memory import ConversationBufferMemory
```
# How the memory is stores?
```
Serialized text — a plain text chat history inserted into prompts.

JSON-like Python objects — internal lists or dicts tracking messages during runtime.

Vectors (embeddings) — stored externally in vector databases for semantic search (like Pinecone or FAISS).
```
