# LangChain Runnables

This repository contains simple examples of different Runnable primitives in LangChain. Runnables are the modern way to connect prompts, LLMs, parsers, retrievers, and custom logic in a clean and flexible way.

## Files

`llm_chains.py`
Demonstrates a basic LLM chain using Groq and a prompt.

`runnable_sequence.py`
Shows how to run multiple steps one after another using RunnableSequence.

`runnable_parallel.py`
Shows how to run multiple tasks at the same time using RunnableParallel.

`runnable_passthrough.py`
Demonstrates how to pass the original input forward without changing it.

`runnable_lambda.py`
Shows how to convert a normal Python function into a Runnable.

`runnable_branch.py`
Demonstrates conditional routing, similar to if-else logic, using RunnableBranch.

`pdf_reader.py`
A simple PDF reader using LangChain and Groq.

`retriever_chain.py`
Shows how to use a retriever to fetch relevant information and answer questions.

## Why Runnables Are Needed

Earlier LangChain had many different chain classes such as LLMChain, RetrievalQA, SequentialChain, and ConversationalRetrievalChain. Each chain had a different structure, which made the framework harder to understand and maintain.

Runnables solve this by giving all components a common interface. Most components can be connected using the pipe operator:

```python
chain = prompt | llm | parser
```

This makes LangChain code cleaner, easier to debug, and more flexible.

